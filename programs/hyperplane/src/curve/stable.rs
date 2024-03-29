//! The stableswap invariant calculator.
use std::convert::TryFrom;

use anchor_lang::{error, Result};
use spl_math::{checked_ceil_div::CheckedCeilDiv, precise_number::PreciseNumber, uint::U256};

use crate::{
    curve::{
        calculator::{
            CurveCalculator, DynAccountSerialize, RoundDirection, SwapWithoutFeesResult,
            TradeDirection, TradingTokenResult,
        },
        math,
    },
    error::SwapError,
    require_msg,
    state::StableCurve,
    try_math,
    utils::math::{AbsDiff, TryMath, TryNew},
};

const N_COINS: u8 = 2;

const ITERATIONS: u16 = 256;

/// Minimum amplification coefficient.
pub const MIN_AMP: u64 = 1;

/// Maximum amplification coefficient.
pub const MAX_AMP: u64 = 1_000_000;

/// Calculates An**n for deriving D
///
/// We choose to use A * n rather than A * n**n because `D**n / prod(x)` loses precision with a huge A value.
fn compute_ann(amp: u64) -> Result<u64> {
    amp.try_mul(N_COINS as u64)
}

/// Returns self to the power of b
fn try_u8_power(a: &U256, b: u8) -> Result<U256> {
    let mut result = *a;
    for _ in 1..b {
        result = try_math!(result.try_mul(*a))?;
    }
    Ok(result)
}

/// Returns self multiplied by b
fn try_u8_mul(a: &U256, b: u8) -> Result<U256> {
    let mut result = *a;
    for _ in 1..b {
        result = try_math!(result.try_add(*a))?;
    }
    Ok(result)
}

/// D = (AnnS + D_P * n) * D / ((Ann - 1) * D + (n + 1) * D_P)
///
/// * `ann` - An**n - Invariant of A - the amplification coefficient times n**(n-1)
/// * `d_init` - Current approximate value of D
/// * `d_product` - Product of all the balances - prod(x/D) // todo - elliot
/// * `sum_x` - sum(x_i) - S - Sum of all the balances
fn compute_next_d(ann: u64, d_init: &U256, d_product: &U256, sum_x: u128) -> Result<U256> {
    // An**n * sum(x)
    let anns = try_math!(U256::from(ann).try_mul(sum_x.into()))?;

    // D = (AnnS + D_P * n) * D / ((Ann - 1) * D + (n + 1) * D_P)
    let numerator = try_math!(anns
        .try_add(try_u8_mul(d_product, N_COINS)?)?
        .try_mul(*d_init))?;
    let denominator = try_math!(d_init
        .try_mul((ann.try_sub(1)?).into())?
        .try_add(try_u8_mul(d_product, N_COINS.try_add(1)?)?))?;

    try_math!(numerator.try_div(denominator))
}

/// Compute stable swap invariant (D)
///
/// Defined as:
///
/// ```md
/// A * sum(x_i) * n**n + D = A * D * n**n + D**(n+1) / (n**n * prod(x_i))
/// ```
/// The value of D is calculated by solving the above equation for D.
///
/// Given all other parameters are constant, f(D), a polynomial function of degree n+1, is represented as:
///
/// ```md
/// f(D) = D**n+1 / n**n * prod(x_i) + (Ann - 1)D - AnnS = 0
///
/// The derivative of the above function is:
///
/// f'(D) = (n + 1) * D_P / D + (Ann - 1)
///
/// Where:
/// - S = sum(x_i)
/// - D_P = D**(n+1) / n**n * prod(x_i)
/// ```
///
/// Solve for D using [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method):
/// ```md
/// Newton's method:
/// x_n+1 = x_n - f(x_n) / f'(x_n)
///
/// Therefore:
/// D_n+1 = D_n - f(D_n) / f'(D_n)
///
/// Iteratively solve for D:
///
/// D = (AnnS + D_P * n) * D / ((Ann - 1) * D + (n + 1) * D_P)
/// ```
///
/// * `ann` - The invariant of A - the amplification coefficient times n**(n-1)
/// * `amount_a` - The number of A tokens in the pool
/// * `amount_b` - The number of B tokens in the pool
fn compute_d(ann: u64, amount_a: u128, amount_b: u128) -> Result<u128> {
    let sum_x = try_math!(amount_a.try_add(amount_b))?; // sum(x_i), a.k.a S
    if sum_x == 0 {
        Ok(0)
    } else {
        let amount_a_times_coins = try_math!(try_u8_mul(&U256::from(amount_a), N_COINS))?;
        let amount_b_times_coins = try_math!(try_u8_mul(&U256::from(amount_b), N_COINS))?;

        let mut d_previous: U256;
        // start by guessing D with the sum(x_i)
        let mut d: U256 = sum_x.into();

        // Iteratively approximate D
        for _ in 0..ITERATIONS {
            // D_P = D**(n+1) / n**n * prod(x_i)
            let mut d_product = d;
            d_product = try_math!(d_product.try_mul(d)?.try_div(amount_a_times_coins))?;
            d_product = try_math!(d_product.try_mul(d)?.try_div(amount_b_times_coins))?;
            d_previous = d;
            // D = (AnnS + D_P * n) * D / ((Ann - 1) * D + (n + 1) * D_P)
            d = try_math!(compute_next_d(ann, &d, &d_product, sum_x))?;

            // Equality with the precision of 1
            if d.abs_diff(d_previous) <= 1.into() {
                break;
            }
        }
        u128::try_from(d).map_err(|_| error!(SwapError::ConversionFailure))
    }
}

/// Compute swap amount `y` in proportion to `x`
///
/// Solve the quadratic equation iteratively for y:
///
/// ```md
///
/// A * sum(x_i) * n**n + D = A * D * n**n + D**(n+1) / (n**n * prod(x_i))
///
/// This forms a polynomial of degree 2 in y:
///
/// f(y) = y**2 + (b - D)y - c = 0
///
/// Where:
/// - b = S + D / Ann
/// - c = D**n+1 / n**n * P * Ann
/// - S = sum(x_i) where i != j
/// - P = prod(x_i) where i != j
///
/// To find the root:
///
/// y_n+1 = y_n - y_n**2 + (b - D)y - c / 2y_n + b - D = y_n**2 + c / 2y_n + b - D
///
/// This can be calculated using Newton's method by iterating:
///
/// y = y**2 + c / 2y + b - D
///
/// The initial value of y is D
/// ```
///
/// * `ann` - A * n**n - Ann - The invariant of A - the amplification coefficient times n**(n-1)
/// * `x` - The number of source tokens in the pool after depositing swap amount
/// * `d` - D - The total amount of tokens when they have an equal price i.e. at equilibrium when all tokens have equal balance
fn compute_y(ann: u64, x: u128, d: u128) -> Result<u128> {
    // Upscale to U256
    let ann: U256 = ann.into();
    let new_source_amount: U256 = x.into();
    let d: U256 = d.into();
    let zero = U256::zero();
    let one = U256::one();

    // b = S + D / Ann
    let b = try_math!(new_source_amount.try_add(d.try_div(ann)?))?;

    // c = D**n+1 / n**n * P * Ann
    // Rewrite this to avoid overflows from D**n+1:
    // c = (D * D / P * n) * (D / Ann * n)
    let mut c = try_math!(d.try_mul(d)?.try_div(x.try_mul(N_COINS.into())?.into()))?;
    c = try_math!(c.try_mul(d)?.try_div(ann.try_mul(N_COINS.into())?))?;

    // Solve for y:
    let mut y = d;
    for _ in 0..ITERATIONS {
        // y = y**2 + c / 2y + b - D
        let numerator = try_math!(try_u8_power(&y, 2)?.try_add(c))?;
        let denominator = try_math!(try_u8_mul(&y, 2)?.try_add(b)?.try_sub(d))?;
        // checked_ceil_div is conservative, not allowing for a 0 return, but we can
        // ceiling to 1 token in this case since we're solving through approximation,
        // and not doing a constant product calculation
        let (y_new, _) = numerator.checked_ceil_div(denominator).unwrap_or_else(|| {
            if numerator == U256::from(0u128) {
                (zero, zero)
            } else {
                (one, zero)
            }
        });
        if y_new == y {
            break;
        } else {
            y = y_new;
        }
    }
    u128::try_from(y).map_err(|_| error!(SwapError::CalculationFailure))
}

fn scale_up(source_amount: u128, factor: u64) -> Result<u128> {
    require_msg!(
        factor > 0,
        SwapError::CalculationFailure,
        "Scaling factor is 0"
    );
    let amount = if factor > 1 {
        try_math!(source_amount.try_mul(factor as u128))?
    } else {
        source_amount
    };
    Ok(amount)
}

fn scale_down(source_amount: u128, factor: u64, round_up: bool) -> Result<u128> {
    require_msg!(
        factor > 0,
        SwapError::CalculationFailure,
        "Scaling factor is 0"
    );
    let amount = if factor > 1 {
        let factor = u128::from(factor);
        let amount = source_amount / factor;
        // Was there any remainder?
        if round_up && factor * amount < source_amount {
            amount + 1
        } else {
            amount
        }
    } else {
        source_amount
    };
    Ok(amount)
}

pub fn scale_pool_inputs(
    curve: &StableCurve,
    source_amount: u128,
    pool_token_a_amount: u128,
    pool_token_b_amount: u128,
    trade_direction: TradeDirection,
) -> Result<(u128, u128, u128)> {
    let pool_token_a_amt_scaled = try_math!(scale_up(pool_token_a_amount, curve.token_a_factor))?;
    let pool_token_b_amt_scaled = try_math!(scale_up(pool_token_b_amount, curve.token_b_factor))?;
    let source_amt_scaled = match trade_direction {
        TradeDirection::AtoB => try_math!(scale_up(source_amount, curve.token_a_factor))?,
        TradeDirection::BtoA => try_math!(scale_up(source_amount, curve.token_b_factor))?,
    };
    Ok((
        source_amt_scaled,
        pool_token_a_amt_scaled,
        pool_token_b_amt_scaled,
    ))
}

pub fn scale_swap_inputs(
    curve: &StableCurve,
    source_amount: u128,
    pool_source_amount: u128,
    pool_destination_amount: u128,
    trade_direction: TradeDirection,
) -> Result<(u128, u128, u128)> {
    let scaled = match trade_direction {
        TradeDirection::AtoB => {
            let source_amt_scaled = try_math!(scale_up(source_amount, curve.token_a_factor))?;
            let pool_source_amt_scaled =
                try_math!(scale_up(pool_source_amount, curve.token_a_factor))?;
            let pool_dest_amt_scaled =
                try_math!(scale_up(pool_destination_amount, curve.token_b_factor))?;
            (
                source_amt_scaled,
                pool_source_amt_scaled,
                pool_dest_amt_scaled,
            )
        }
        TradeDirection::BtoA => {
            let source_amt_scaled = try_math!(scale_up(source_amount, curve.token_b_factor))?;
            let pool_source_amt_scaled =
                try_math!(scale_up(pool_source_amount, curve.token_b_factor))?;
            let pool_dest_amt_scaled =
                try_math!(scale_up(pool_destination_amount, curve.token_a_factor))?;
            (
                source_amt_scaled,
                pool_source_amt_scaled,
                pool_dest_amt_scaled,
            )
        }
    };
    Ok(scaled)
}

pub fn scale_swap_outputs(
    curve: &StableCurve,
    new_pool_destination_amount: u128,
    trade_direction: TradeDirection,
) -> Result<u128> {
    let factor = match trade_direction {
        TradeDirection::AtoB => curve.token_b_factor,
        TradeDirection::BtoA => curve.token_a_factor,
    };
    let new_pool_destination_amount = try_math!(scale_down(
        new_pool_destination_amount,
        factor,
        true // round up to ensure the pool is favoured
    ))?;
    Ok(new_pool_destination_amount)
}

impl CurveCalculator for StableCurve {
    /// Stable curve
    fn swap_without_fees(
        &self,
        source_amount: u128,
        pool_source_amount: u128,
        pool_destination_amount: u128,
        trade_direction: TradeDirection,
    ) -> Result<SwapWithoutFeesResult> {
        if source_amount == 0 {
            return Ok(SwapWithoutFeesResult {
                source_amount_swapped: 0,
                destination_amount_swapped: 0,
            });
        }
        let ann = compute_ann(self.amp)?;

        let (source_amt_scaled, pool_source_amt_scaled, pool_dest_amt_scaled) =
            try_math!(scale_swap_inputs(
                self,
                source_amount,
                pool_source_amount,
                pool_destination_amount,
                trade_direction,
            ))?;

        let new_source_amount = try_math!(pool_source_amt_scaled.try_add(source_amt_scaled))?;
        let new_destination_amount = try_math!(compute_y(
            ann,
            new_source_amount,
            try_math!(compute_d(ann, pool_source_amt_scaled, pool_dest_amt_scaled))?,
        ))?;

        let amount_swapped = try_math!(pool_destination_amount.try_sub(scale_swap_outputs(
            self,
            new_destination_amount,
            trade_direction
        )?))?;

        Ok(SwapWithoutFeesResult {
            source_amount_swapped: source_amount,
            destination_amount_swapped: amount_swapped,
        })
    }

    /// Remove pool tokens from the pool in exchange for trading tokens
    /// Returns the amounts of trading tokens that were redeemed
    /// * `pool_tokens` - the amount of pool tokens to burn
    /// * `pool_token_supply` - the total supply of pool tokens
    /// * `pool_token_a_amount` - the amount of token A in the pool
    /// * `pool_token_b_amount` - the amount of token B in the pool
    /// * `round_direction` - the direction to round the output trading token amounts
    fn pool_tokens_to_trading_tokens(
        &self,
        pool_tokens: u128,
        pool_token_supply: u128,
        pool_token_a_amount: u128,
        pool_token_b_amount: u128,
        round_direction: RoundDirection,
    ) -> Result<TradingTokenResult> {
        math::pool_tokens_to_trading_tokens(
            pool_tokens,
            pool_token_supply,
            pool_token_a_amount,
            pool_token_b_amount,
            round_direction,
        )
    }

    fn validate(&self) -> Result<()> {
        require_msg!(
            self.amp > MIN_AMP,
            SwapError::InvalidCurve,
            &format!("amp={} <= MIN_AMP={}", self.amp, MIN_AMP)
        );
        require_msg!(
            self.amp < MAX_AMP,
            SwapError::InvalidCurve,
            &format!("amp={} >= MAX_AMP={}", self.amp, MAX_AMP)
        );

        Ok(())
    }

    fn normalized_value(
        &self,
        pool_token_a_amount: u128,
        pool_token_b_amount: u128,
    ) -> Result<PreciseNumber> {
        #[cfg(not(any(test, feature = "fuzz")))]
        {
            let leverage = compute_ann(self.amp)?;
            PreciseNumber::try_new(compute_d(
                leverage,
                pool_token_a_amount,
                pool_token_b_amount,
            )?)
        }
        #[cfg(any(test, feature = "fuzz"))]
        {
            use roots::{find_roots_cubic_normalized, Roots};

            use crate::utils::math::TryMathRef;

            let x = pool_token_a_amount as f64;
            let y = pool_token_b_amount as f64;
            let c = (4.0 * (self.amp as f64)) - 1.0;
            let d = 16.0 * (self.amp as f64) * x * y * (x + y);
            let roots = find_roots_cubic_normalized(0.0, c, d);
            let x0 = match roots {
                Roots::No(_) => panic!("No roots found for cubic equations"),
                Roots::One(x) => x[0],
                Roots::Two(_) => panic!("Two roots found for cubic, mathematically impossible"),
                Roots::Three(x) => x[1],
                Roots::Four(_) => panic!("Four roots found for cubic, mathematically impossible"),
            };

            let root_uint = (x0 * ((10f64).powf(11.0))).round() as u128;
            let precision = PreciseNumber::try_new(10)?.try_pow(11)?;
            let two = PreciseNumber::try_new(2)?;
            PreciseNumber::try_new(root_uint)?
                .try_div(&precision)?
                .try_div(&two)
        }
    }
}

impl DynAccountSerialize for StableCurve {
    fn try_dyn_serialize(&self, mut dst: std::cell::RefMut<&mut [u8]>) -> Result<()> {
        let dst: &mut [u8] = &mut dst;
        let mut cursor = std::io::Cursor::new(dst);
        anchor_lang::AccountSerialize::try_serialize(self, &mut cursor)
    }
}

#[cfg(test)]
mod tests {

    // useful for d.p. clarity in tests
    #![allow(clippy::inconsistent_digit_grouping)]

    use std::{borrow::BorrowMut, cmp::Ordering};

    use anchor_lang::AccountDeserialize;
    use hyperplane_sim::StableSwapModel;
    use proptest::prelude::*;

    use super::*;
    use crate::{
        curve::calculator::{
            test::{
                check_curve_value_from_swap, check_pool_value_from_deposit,
                check_pool_value_from_withdraw, total_and_intermediate,
            },
            RoundDirection, INITIAL_SWAP_POOL_AMOUNT,
        },
        state::Curve,
        utils::math::decimals_to_factor,
    };

    #[test]
    fn initial_pool_amount() {
        let amp = 1;
        let calculator = StableCurve {
            amp,
            token_a_factor: 1,
            token_b_factor: 1,
            ..Default::default()
        };
        assert_eq!(calculator.new_pool_supply(), INITIAL_SWAP_POOL_AMOUNT);
    }

    fn check_pool_token_rate(
        token_a: u128,
        token_b: u128,
        deposit: u128,
        supply: u128,
        expected_a: u128,
        expected_b: u128,
    ) {
        let amp = 1;
        let calculator = StableCurve {
            amp,
            token_a_factor: 0,
            token_b_factor: 0,
            ..Default::default()
        };
        let results = calculator
            .pool_tokens_to_trading_tokens(
                deposit,
                supply,
                token_a,
                token_b,
                RoundDirection::Ceiling,
            )
            .unwrap();
        assert_eq!(results.token_a_amount, expected_a);
        assert_eq!(results.token_b_amount, expected_b);
    }

    // rounding up
    #[test]
    fn trading_token_conversion() {
        check_pool_token_rate(2, 49, 5, 10, 1, 25);
        check_pool_token_rate(100, 202, 5, 101, 5, 10);
        check_pool_token_rate(5, 501, 2, 10, 1, 101);
    }

    #[test]
    fn swap_zero() {
        let curve = StableCurve {
            amp: 100,
            token_a_factor: 1,
            token_b_factor: 1,
            ..Default::default()
        };
        let result = curve.swap_without_fees(0, 100, 1_000_000_000_000_000, TradeDirection::AtoB);

        let result = result.unwrap();
        assert_eq!(result.source_amount_swapped, 0);
        assert_eq!(result.destination_amount_swapped, 0);
    }

    #[test]
    fn serialize_stable_curve() {
        let amp = u64::MAX;
        let curve = StableCurve {
            amp,
            token_a_factor: 1,
            token_b_factor: 1,
            ..Default::default()
        };

        let mut arr = [0u8; Curve::LEN];
        let packed = arr.borrow_mut();
        let ref_mut = std::cell::RefCell::new(packed);

        curve.try_dyn_serialize(ref_mut.borrow_mut()).unwrap();
        let unpacked = StableCurve::try_deserialize(&mut arr.as_ref()).unwrap();
        assert_eq!(curve, unpacked);
    }

    proptest! {
        #[test]
        fn curve_value_does_not_decrease_from_deposit(
            pool_token_amount in 1..u64::MAX as u128,
            pool_token_supply in 1..u64::MAX as u128,
            swap_token_a_amount in 1..u64::MAX as u128,
            swap_token_b_amount in 1..u64::MAX as u128,
            amp in MIN_AMP..MAX_AMP,
            token_a_decimals in 5..12_u8,
            token_b_decimals in 5..12_u8,
        ) {
            // Make sure we will get at least one trading token out for each
            // side, otherwise the calculation fails
            prop_assume!(pool_token_amount * swap_token_a_amount / pool_token_supply >= 1);
            prop_assume!(pool_token_amount * swap_token_b_amount / pool_token_supply >= 1);

            let curve = StableCurve::new(amp, token_a_decimals, token_b_decimals).unwrap();

            check_pool_value_from_deposit(
                &curve,
                pool_token_amount,
                pool_token_supply,
                swap_token_a_amount,
                swap_token_b_amount,
            );
        }
    }

    proptest! {
        #[test]
        fn curve_value_does_not_decrease_from_withdraw(
            (pool_token_supply, pool_token_amount) in total_and_intermediate(u64::MAX),
            swap_token_a_amount in 1..u64::MAX as u128,
            swap_token_b_amount in 1..u64::MAX as u128,
            amp in MIN_AMP..MAX_AMP,
            token_a_decimals in 5..12_u8,
            token_b_decimals in 5..12_u8,
        ) {
            let pool_token_amount = pool_token_amount as u128;
            let pool_token_supply = pool_token_supply as u128;
            // Make sure we will get at least one trading token out for each
            // side, otherwise the calculation fails
            prop_assume!(pool_token_amount * swap_token_a_amount / pool_token_supply >= 1);
            prop_assume!(pool_token_amount * swap_token_b_amount / pool_token_supply >= 1);

            let curve = StableCurve::new(amp, token_a_decimals, token_b_decimals).unwrap();

            check_pool_value_from_withdraw(
                &curve,
                pool_token_amount,
                pool_token_supply,
                swap_token_a_amount,
                swap_token_b_amount,
            );
        }
    }

    proptest! {
        #[test]
        fn curve_value_does_not_decrease_from_swap(
            source_token_amount in 1..u64::MAX,
            swap_source_amount in 1..u64::MAX,
            swap_destination_amount in 1..u64::MAX,
            amp in MIN_AMP..MAX_AMP,
            token_a_decimals in 5..12_u8,
            token_b_decimals in 5..12_u8,
        ) {
            let curve = StableCurve::new(amp, token_a_decimals, token_b_decimals).unwrap();

            check_curve_value_from_swap(
                &curve,
                source_token_amount as u128,
                swap_source_amount as u128,
                swap_destination_amount as u128,
                TradeDirection::AtoB
            );
        }
    }

    // Test to compare pools of scaled values vs a 6 d.p. / 6 d.p. unscaled pool
    proptest! {
        #[test]
        fn compare_swap_pool_with_different_decimals(
            source_token_amount in 1..u64::MAX as u128,
            pool_source_amount in 1..u64::MAX as u128,
            pool_destination_amount in 1..u64::MAX as u128,
            amp in MIN_AMP..MAX_AMP,
            token_a_decimals in 6..12_u8,
            token_b_decimals in 6..12_u8,
        ) {
            // can't have more source tokens than the token program allows
            prop_assume!(source_token_amount + pool_source_amount <= u64::MAX as u128);
            // we are comparing against a pool with 6 dp / 6 dp tokens
            prop_assume!(token_a_decimals != 6 && token_b_decimals != 6);

            // create a pool with 6 dp for tokens A + B
            let stable_curve = StableCurve::new(amp, 6, 6).unwrap();

            // create a pool with random dp for tokens A + B
            let stable_curve_mismatched = StableCurve::new(amp, token_a_decimals, token_b_decimals).unwrap();

            // swap the same amount of tokens in both pools
            let stable_result = stable_curve.swap_without_fees(
                source_token_amount,
                pool_source_amount,
                pool_destination_amount,
                TradeDirection::AtoB
            ).unwrap();

            // perform the same swap, but scale the amounts from 6 dp to the pool's token decimals
            let stable_mismatched_result = stable_curve_mismatched.swap_without_fees(
                scale_decimal(source_token_amount, 6, token_a_decimals, false),
                scale_decimal(pool_source_amount, 6, token_a_decimals, false),
                scale_decimal(pool_destination_amount, 6, token_b_decimals, false),
                TradeDirection::AtoB
            ).unwrap();

            // scale the result back to 6 dp
            let stable_mismatched_result = SwapWithoutFeesResult {
                source_amount_swapped: scale_decimal(stable_mismatched_result.source_amount_swapped, token_a_decimals, 6, true),
                destination_amount_swapped: scale_decimal(stable_mismatched_result.destination_amount_swapped, token_b_decimals, 6, false),
            };

            assert_eq!(stable_result.source_amount_swapped, stable_mismatched_result.source_amount_swapped);

            // The decimals mismatch result can be off by 1 in either direction
            assert!(stable_result.destination_amount_swapped.abs_diff(stable_mismatched_result.destination_amount_swapped) <= 1,
                "\nstable_result.destination_amount_swapped:\n {}\nstable_mismatched_result.destination_amount_swapped:\n {}\n",
                stable_result.destination_amount_swapped,
                stable_mismatched_result.destination_amount_swapped
            );
        }
    }

    fn scale_decimal(amount: u128, current_decimals: u8, new_decimals: u8, round_up: bool) -> u128 {
        match current_decimals.cmp(&new_decimals) {
            Ordering::Greater => {
                let factor = 10_u128.pow((current_decimals - new_decimals) as u32);
                let amt = amount / factor;
                if round_up && amount % factor > 0 {
                    amt + 1
                } else {
                    amt
                }
            }
            Ordering::Less => amount * 10_u128.pow((new_decimals - current_decimals) as u32),
            Ordering::Equal => amount,
        }
    }

    const F5: u128 = 10_u128.pow(5);
    const F6: u128 = 10_u128.pow(6);
    const F7: u128 = 10_u128.pow(7);
    const F8: u128 = 10_u128.pow(8);
    const F9: u128 = 10_u128.pow(8);
    const F10: u128 = 10_u128.pow(10);
    const F11: u128 = 10_u128.pow(11);
    const F12: u128 = 10_u128.pow(12);

    #[derive(Debug, Clone, Copy)]
    struct SwapTestCase {
        amp: u64,
        token_a_decimals: u8,
        token_b_decimals: u8,
        source_token_amount: u128,
        pool_source_amount: u128,
        pool_destination_amount: u128,
        expected_source_amount_swapped: u128,
        expected_destination_amount_swapped: u128,
    }

    fn check_swap(
        SwapTestCase {
            amp,
            token_a_decimals,
            token_b_decimals,
            source_token_amount,
            pool_source_amount,
            pool_destination_amount,
            expected_source_amount_swapped,
            expected_destination_amount_swapped,
        }: SwapTestCase,
    ) {
        let curve = StableCurve::new(amp, token_a_decimals, token_b_decimals).unwrap();

        let results = curve
            .swap_without_fees(
                source_token_amount,
                pool_source_amount,
                pool_destination_amount,
                TradeDirection::AtoB,
            )
            .unwrap();

        assert_eq!(
            results.source_amount_swapped,
            expected_source_amount_swapped
        );
        assert_eq!(
            results.destination_amount_swapped,
            expected_destination_amount_swapped
        );
    }

    #[test]
    fn run_swap_scenarios() {
        {
            check_swap(SwapTestCase {
                amp: 75,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 924_745,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 934_112,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 978_133,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 10_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 992_978,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 997_768,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 999_293,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 10_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 999_776,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 999_929,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 1_000_000,
                expected_destination_amount_swapped: 999_977,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 10_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 10_000_000,
                expected_destination_amount_swapped: 999_999,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 100_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 100_000_000,
                expected_destination_amount_swapped: 999_999,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 10_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1,
                expected_source_amount_swapped: 10_000_000,
                expected_destination_amount_swapped: 0,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 10_000_000,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1,
                expected_source_amount_swapped: 10_000_000,
                expected_destination_amount_swapped: 0,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000_000_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1,
                pool_source_amount: 1_000_000,
                pool_destination_amount: 1,
                expected_source_amount_swapped: 1,
                expected_destination_amount_swapped: 0,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 100,
                pool_source_amount: 100_000,
                pool_destination_amount: 127_700,
                expected_source_amount_swapped: 100,
                expected_destination_amount_swapped: 113,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 10,
                pool_source_amount: 10_000,
                pool_destination_amount: 12_770,
                expected_source_amount_swapped: 10,
                expected_destination_amount_swapped: 11,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1,
                pool_source_amount: 1_000,
                pool_destination_amount: 1_277,
                expected_source_amount_swapped: 1,
                expected_destination_amount_swapped: 1,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1,
                pool_source_amount: 127,
                pool_destination_amount: 127,
                expected_source_amount_swapped: 1,
                expected_destination_amount_swapped: 0,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 10,
                pool_source_amount: 1_000,
                pool_destination_amount: 1_000_000,
                expected_source_amount_swapped: 10,
                expected_destination_amount_swapped: 9_999,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 8,
                source_token_amount: 10,
                pool_source_amount: 1_000,
                pool_destination_amount: 100_000,
                expected_source_amount_swapped: 10,
                expected_destination_amount_swapped: 999, // 999.9 rounded down in favour of pool
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 7,
                source_token_amount: 10,
                pool_source_amount: 1_000,
                pool_destination_amount: 10_000,
                expected_source_amount_swapped: 10,
                expected_destination_amount_swapped: 99, // 99.99 rounded down in favour of pool
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 10,
                pool_source_amount: 1_000,
                pool_destination_amount: 1_000,
                expected_source_amount_swapped: 10,
                expected_destination_amount_swapped: 9, // 9.999 rounded down in favour of pool
            });
        }
        {
            // 5 dp is bad - you get 0 for 10 lamports even though the pool is balanced
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 5,
                source_token_amount: 10,
                pool_source_amount: 1_000,
                pool_destination_amount: 100,
                expected_source_amount_swapped: 10,
                expected_destination_amount_swapped: 0, // 0.9999 rounded down in favour of pool
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 1,
                pool_source_amount: 2,
                pool_destination_amount: 1,
                expected_source_amount_swapped: 1,
                expected_destination_amount_swapped: 0, // rounded down in favour of pool
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 10_000 * F6,
                pool_source_amount: 1_000_000 * F6,
                pool_destination_amount: 1_000_000 * F9,
                expected_source_amount_swapped: 10_000 * F6,
                expected_destination_amount_swapped: 8_646_023887918,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 10_000 * F6,
                pool_source_amount: 1_000_000 * F6,
                pool_destination_amount: 1_000_000 * F9,
                expected_source_amount_swapped: 10_000 * F6,
                expected_destination_amount_swapped: 9_837_239410620,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 100,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 10_000 * F6,
                pool_source_amount: 5_000_000 * F6,
                pool_destination_amount: 2_500_000 * F9,
                expected_source_amount_swapped: 10_000 * F6,
                expected_destination_amount_swapped: 6_597_668227728,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1000,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 10_000 * F6,
                pool_source_amount: 5_000_000 * F6,
                pool_destination_amount: 2_500_000 * F9,
                expected_source_amount_swapped: 10_000 * F6,
                expected_destination_amount_swapped: 9_464_275425005,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F9,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 977_451_470791890,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 10_000,
                token_a_decimals: 6,
                token_b_decimals: 5,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F5,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 999_976_98198,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 10_000,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F6,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 999_976_981984,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 10_000,
                token_a_decimals: 6,
                token_b_decimals: 7,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F7,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 999_976_9819850,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 10_000,
                token_a_decimals: 6,
                token_b_decimals: 8,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F8,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 999_976_98198500,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1_000,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F9,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 977_451_470791890,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 10_000,
                token_a_decimals: 6,
                token_b_decimals: 9,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F9,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 997_684_910256419,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1000,
                token_a_decimals: 6,
                token_b_decimals: 10,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F10,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 999_770_0601323903,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1000,
                token_a_decimals: 6,
                token_b_decimals: 11,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F11,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 999_770_06013239031,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 1000,
                token_a_decimals: 6,
                token_b_decimals: 12,
                source_token_amount: 1_000_000 * F6,
                pool_source_amount: 500_000_000 * F6,
                pool_destination_amount: 400_000_000 * F12,
                expected_source_amount_swapped: 1_000_000 * F6,
                expected_destination_amount_swapped: 999_770_060132390312,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 500,
                token_a_decimals: 9,
                token_b_decimals: 9,
                source_token_amount: 1,
                pool_source_amount: 100_000_000 * F9,
                pool_destination_amount: 100_000_999 * F9,
                expected_source_amount_swapped: 1,
                expected_destination_amount_swapped: 1,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 50,
                token_a_decimals: 9,
                token_b_decimals: 9,
                source_token_amount: 1,
                pool_source_amount: 100_000_000 * F9,
                pool_destination_amount: 100_000_999 * F9,
                expected_source_amount_swapped: 1,
                expected_destination_amount_swapped: 1,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 5,
                token_a_decimals: 9,
                token_b_decimals: 9,
                source_token_amount: 1,
                pool_source_amount: 100_000_000 * F9,
                pool_destination_amount: 100_000_999 * F9,
                expected_source_amount_swapped: 1,
                expected_destination_amount_swapped: 1,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 19,
                token_a_decimals: 6,
                token_b_decimals: 6,
                source_token_amount: 1_081_921_530_148_278_930,
                pool_source_amount: 8_428_871_396_984_204005,
                pool_destination_amount: 9_377_337_809_125_992025,
                expected_source_amount_swapped: 1_081_921_530_148_278930,
                expected_destination_amount_swapped: 1_081_107_876_398_213723,
            });
        }
        {
            check_swap(SwapTestCase {
                amp: 19,
                token_a_decimals: 7,
                token_b_decimals: 7,
                source_token_amount: 108_192_153_014_8278930,
                pool_source_amount: 842_887_139_698_4204005,
                pool_destination_amount: 937_733_780_912_5992025,
                expected_source_amount_swapped: 108_192_153_014_8278930,
                expected_destination_amount_swapped: 108_110_787_639_8213723,
            });
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct WithdrawTestCase {
        pool_token_a_amount: u128,
        pool_token_b_amount: u128,
        pool_token_amount: u128,
        pool_token_supply: u128,
        expected_token_a: u128,
        expected_token_b: u128,
    }

    fn check_withdraw(
        WithdrawTestCase {
            pool_token_a_amount,
            pool_token_b_amount,
            pool_token_amount,
            pool_token_supply,
            expected_token_a,
            expected_token_b,
        }: WithdrawTestCase,
    ) {
        let curve = StableCurve::new(100, 6, 6).unwrap();

        let results = curve
            .pool_tokens_to_trading_tokens(
                pool_token_amount,
                pool_token_supply,
                pool_token_a_amount,
                pool_token_b_amount,
                RoundDirection::Floor,
            )
            .unwrap();

        assert_eq!(results.token_a_amount, expected_token_a);
        assert_eq!(results.token_b_amount, expected_token_b);
    }

    #[test]
    fn run_withdraw_scenarios() {
        {
            check_withdraw(WithdrawTestCase {
                pool_token_amount: 100_000_000 * F6,
                pool_token_supply: 100_000_000 * F6,
                pool_token_a_amount: 100_000_000 * F6,
                pool_token_b_amount: 100_000_000 * F6,
                expected_token_a: 100_000_000 * F6,
                expected_token_b: 100_000_000 * F6,
            });
        }
        {
            check_withdraw(WithdrawTestCase {
                pool_token_amount: 1_000_000 * F6,
                pool_token_supply: 10_000_000 * F6,
                pool_token_a_amount: 100_000_000 * F12,
                pool_token_b_amount: 100_000_000 * F6,
                expected_token_a: 10_000_000 * F12,
                expected_token_b: 10_000_000 * F6,
            });
        }
        {
            check_withdraw(WithdrawTestCase {
                pool_token_amount: 1,
                pool_token_supply: 100,
                pool_token_a_amount: 999,
                pool_token_b_amount: 1,
                expected_token_a: 9,
                expected_token_b: 0,
            });
        }
        {
            check_withdraw(WithdrawTestCase {
                pool_token_amount: 5,
                pool_token_supply: 10,
                pool_token_a_amount: 2,
                pool_token_b_amount: 49,
                expected_token_a: 1,
                expected_token_b: 24, // rounding down
            });
        }
        {
            check_withdraw(WithdrawTestCase {
                pool_token_amount: 5,
                pool_token_supply: 101,
                pool_token_a_amount: 100,
                pool_token_b_amount: 202,
                expected_token_a: 4, // rounding down
                expected_token_b: 10,
            });
        }
        {
            check_withdraw(WithdrawTestCase {
                pool_token_amount: 2,
                pool_token_supply: 10,
                pool_token_a_amount: 5,
                pool_token_b_amount: 501,
                expected_token_a: 1,
                expected_token_b: 100, // rounding down
            });
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct DepositTestCase {
        pool_token_a_amount: u128,
        pool_token_b_amount: u128,
        pool_token_amount: u128,
        pool_token_supply: u128,
        expected_token_a: u128,
        expected_token_b: u128,
    }

    fn check_deposit(
        DepositTestCase {
            pool_token_a_amount,
            pool_token_b_amount,
            pool_token_amount,
            pool_token_supply,
            expected_token_a,
            expected_token_b,
        }: DepositTestCase,
    ) {
        let curve = StableCurve::new(100, 6, 6).unwrap();

        let results = curve
            .pool_tokens_to_trading_tokens(
                pool_token_amount,
                pool_token_supply,
                pool_token_a_amount,
                pool_token_b_amount,
                RoundDirection::Ceiling,
            )
            .unwrap();

        assert_eq!(results.token_a_amount, expected_token_a);
        assert_eq!(results.token_b_amount, expected_token_b);
    }

    #[test]
    fn run_deposit_scenarios() {
        {
            check_deposit(DepositTestCase {
                pool_token_amount: 100_000_000 * F6,
                pool_token_supply: 100_000_000 * F6,
                pool_token_a_amount: 100_000_000 * F6,
                pool_token_b_amount: 100_000_000 * F6,
                expected_token_a: 100_000_000 * F6,
                expected_token_b: 100_000_000 * F6,
            });
        }
        {
            check_deposit(DepositTestCase {
                pool_token_amount: 1_000_000 * F6,
                pool_token_supply: 10_000_000 * F6,
                pool_token_a_amount: 100_000_000 * F12,
                pool_token_b_amount: 100_000_000 * F6,
                expected_token_a: 10_000_000 * F12,
                expected_token_b: 10_000_000 * F6,
            });
        }
        {
            check_deposit(DepositTestCase {
                pool_token_amount: 1,
                pool_token_supply: 100,
                pool_token_a_amount: 999,
                pool_token_b_amount: 1,
                expected_token_a: 10, // rounding up
                expected_token_b: 0,
            });
        }
        {
            check_deposit(DepositTestCase {
                pool_token_amount: 5,
                pool_token_supply: 10,
                pool_token_a_amount: 2,
                pool_token_b_amount: 49,
                expected_token_a: 1,
                expected_token_b: 25, // rounding up
            });
        }
        {
            check_deposit(DepositTestCase {
                pool_token_amount: 5,
                pool_token_supply: 101,
                pool_token_a_amount: 100,
                pool_token_b_amount: 202,
                expected_token_a: 5, // rounding up
                expected_token_b: 10,
            });
        }
        {
            check_deposit(DepositTestCase {
                pool_token_amount: 2,
                pool_token_supply: 10,
                pool_token_a_amount: 5,
                pool_token_b_amount: 501,
                expected_token_a: 1,
                expected_token_b: 101, // rounding up
            });
        }
    }

    proptest! {
        #[test]
        fn compare_sim_swap_no_fee(
            swap_source_amount in 100..1_000_000_000_000_000_000_u128,
            swap_destination_amount in 100..1_000_000_000_000_000_000_u128,
            source_amount in 100..100_000_000_000_u128,
            amp in MIN_AMP..MAX_AMP,
            token_a_decimals in 5..12_u8,
            token_b_decimals in 5..12_u8,
        ) {
            prop_assume!(source_amount < swap_source_amount);

            let curve = StableCurve::new(amp, token_a_decimals, token_b_decimals).unwrap();

            let mut model: StableSwapModel = StableSwapModel::new(
                amp.into(),
                vec![swap_source_amount, swap_destination_amount],
                vec![decimals_to_factor(token_a_decimals, token_b_decimals).unwrap().into(), decimals_to_factor(token_b_decimals, token_a_decimals).unwrap().into()],
                N_COINS,
            );

            let result = curve.swap_without_fees(
                source_amount,
                swap_source_amount,
                swap_destination_amount,
                TradeDirection::AtoB,
            );

            let result = result.unwrap();
            let sim_result = model.sim_exchange(0, 1, source_amount);

            let diff = sim_result.abs_diff(result.destination_amount_swapped);

            // tolerate a difference of 2 because of the ceiling during calculation
            let tolerance = std::cmp::max(2, sim_result / 1_000_000_000);

            assert!(
                diff <= tolerance,
                "result={}, sim_result={}, diff={}, amp={}, token_a_decimals={}, token_b_decimals={}, source_amount={}, swap_source_amount={}, swap_destination_amount={}",
                result.destination_amount_swapped,
                sim_result,
                diff,
                amp,
                token_a_decimals,
                token_b_decimals,
                source_amount,
                swap_source_amount,
                swap_destination_amount,
            );
        }
    }
}
