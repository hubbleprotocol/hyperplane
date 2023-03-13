//! Base curve implementation

use std::{fmt::Debug, sync::Arc};

use anchor_lang::Result;
#[cfg(feature = "fuzz")]
use arbitrary::Arbitrary;
use num_enum::{IntoPrimitive, TryFromPrimitive};

use crate::{
    curve::{
        calculator::{CurveCalculator, RoundDirection, SwapWithoutFeesResult, TradeDirection},
        fees::Fees,
    },
    model::CurveParameters,
    state::{ConstantPriceCurve, ConstantProductCurve, OffsetCurve, StableCurve},
    try_math,
    utils::math::TryMath,
};

/// Curve types supported by the hyperplane program.
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, IntoPrimitive, TryFromPrimitive)]
pub enum CurveType {
    /// Uniswap-style constant product curve, invariant = token_a_amount * token_b_amount
    ConstantProduct = 1,
    /// Flat line, always providing 1:1 from one token to another
    ConstantPrice = 2,
    /// Offset curve, like Uniswap, but the token B side has a faked offset
    Offset = 3,
    /// Stable curve, like constant product with less slippage around a fixed price
    Stable = 4,
}

/// Encodes all results of swapping from a source token to a destination token
#[derive(Debug, PartialEq)]
pub struct SwapResult {
    /// New amount of source token
    pub new_swap_source_amount: u128,
    /// New amount of destination token
    pub new_swap_destination_amount: u128,
    /// Amount of source token swapped (includes fees)
    pub source_amount_swapped: u128,
    /// Amount of destination token swapped
    pub destination_amount_swapped: u128,
    /// Amount of source tokens going to pool holders
    pub trade_fee: u128,
    /// Amount of source tokens going to owner
    pub owner_fee: u128,
}

/// Concrete struct to wrap around the trait object which performs calculation.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SwapCurve {
    /// The type of curve contained in the calculator, helpful for outside
    /// queries
    pub curve_type: CurveType,
    /// The actual calculator, represented as a trait object to allow for many
    /// different types of curves
    pub calculator: Arc<dyn CurveCalculator + Sync + Send>,
}

impl SwapCurve {
    pub fn new_from_params(curve_params: CurveParameters) -> Result<Self> {
        let curve = match curve_params {
            CurveParameters::ConstantProduct => SwapCurve {
                curve_type: CurveType::ConstantProduct,
                calculator: Arc::new(ConstantProductCurve {
                    ..Default::default()
                }),
            },
            CurveParameters::ConstantPrice { token_b_price } => SwapCurve {
                curve_type: CurveType::ConstantPrice,
                calculator: Arc::new(ConstantPriceCurve {
                    token_b_price,
                    ..Default::default()
                }),
            },
            CurveParameters::Offset { token_b_offset } => SwapCurve {
                curve_type: CurveType::Offset,
                calculator: Arc::new(OffsetCurve {
                    token_b_offset,
                    ..Default::default()
                }),
            },
            CurveParameters::Stable {
                amp,
                token_a_decimals,
                token_b_decimals,
            } => SwapCurve {
                curve_type: CurveType::Stable,
                calculator: Arc::new(StableCurve::new(amp, token_a_decimals, token_b_decimals)?),
            },
        };
        Ok(curve)
    }

    /// Subtract fees and calculate how much destination token will be provided
    /// given an amount of source token.
    pub fn swap(
        &self,
        source_amount: u128,
        swap_source_amount: u128,
        swap_destination_amount: u128,
        trade_direction: TradeDirection,
        fees: &Fees,
    ) -> Result<SwapResult> {
        // debit the fee to calculate the amount swapped
        let trade_fee = try_math!(fees.trading_fee(source_amount))?;
        let owner_fee = try_math!(fees.owner_trading_fee(source_amount))?;

        let total_fees = try_math!(trade_fee.try_add(owner_fee))?;
        let source_amount_less_fees = try_math!(source_amount.try_sub(total_fees))?;

        let SwapWithoutFeesResult {
            source_amount_swapped,
            destination_amount_swapped,
        } = self.calculator.swap_without_fees(
            source_amount_less_fees,
            swap_source_amount,
            swap_destination_amount,
            trade_direction,
        )?;

        let source_amount_swapped = try_math!(source_amount_swapped.try_add(total_fees))?;
        Ok(SwapResult {
            new_swap_source_amount: try_math!(swap_source_amount.try_add(source_amount_swapped))?,
            new_swap_destination_amount: try_math!(
                swap_destination_amount.try_sub(destination_amount_swapped)
            )?,
            source_amount_swapped,
            destination_amount_swapped,
            trade_fee,
            owner_fee,
        })
    }

    /// Get the amount of pool tokens for the deposited amount of token A or B
    pub fn deposit_single_token_type(
        &self,
        source_amount: u128,
        swap_token_a_amount: u128,
        swap_token_b_amount: u128,
        pool_supply: u128,
        trade_direction: TradeDirection,
        fees: &Fees,
    ) -> Result<u128> {
        if source_amount == 0 {
            return Ok(0);
        }
        // Get the trading fee incurred if *half* the source amount is swapped
        // for the other side. Reference at:
        // https://github.com/balancer-labs/balancer-core/blob/f4ed5d65362a8d6cec21662fb6eae233b0babc1f/contracts/BMath.sol#L117
        let half_source_amount = std::cmp::max(1, try_math!(source_amount.try_div(2))?);
        let trade_fee = try_math!(fees.trading_fee(half_source_amount))?;
        let owner_fee = try_math!(fees.owner_trading_fee(half_source_amount))?;
        let total_fees = try_math!(trade_fee.try_add(owner_fee))?;
        let source_amount = try_math!(source_amount.try_sub(total_fees))?;
        self.calculator.deposit_single_token_type(
            source_amount,
            swap_token_a_amount,
            swap_token_b_amount,
            pool_supply,
            trade_direction,
        )
    }

    /// Get the amount of pool tokens for the withdrawn amount of token A or B
    pub fn withdraw_single_token_type_exact_out(
        &self,
        source_amount: u128,
        swap_token_a_amount: u128,
        swap_token_b_amount: u128,
        pool_supply: u128,
        trade_direction: TradeDirection,
        fees: &Fees,
    ) -> Result<u128> {
        if source_amount == 0 {
            return Ok(0);
        }
        // Since we want to get the amount required to get the exact amount out,
        // we need the inverse trading fee incurred if *half* the source amount
        // is swapped for the other side. Reference at:
        // https://github.com/balancer-labs/balancer-core/blob/f4ed5d65362a8d6cec21662fb6eae233b0babc1f/contracts/BMath.sol#L117
        let half_source_amount = try_math!(source_amount.try_add(1)?.try_div(2))?; // round up
        let pre_fee_source_amount = try_math!(fees.pre_trading_fee_amount(half_source_amount))?;
        let source_amount = try_math!(source_amount
            .try_sub(half_source_amount)?
            .try_add(pre_fee_source_amount))?;
        self.calculator.withdraw_single_token_type_exact_out(
            source_amount,
            swap_token_a_amount,
            swap_token_b_amount,
            pool_supply,
            trade_direction,
            RoundDirection::Ceiling,
        )
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;

    use super::*;
    use crate::curve::calculator::test::total_and_intermediate;

    #[test]
    fn constant_product_trade_fee() {
        // calculation on https://github.com/solana-labs/solana-program-library/issues/341
        let swap_source_amount = 1000;
        let swap_destination_amount = 50000;
        let trade_fee_numerator = 1;
        let trade_fee_denominator = 100;
        let owner_trade_fee_numerator = 0;
        let owner_trade_fee_denominator = 0;
        let owner_withdraw_fee_numerator = 0;
        let owner_withdraw_fee_denominator = 0;
        let host_fee_numerator = 0;
        let host_fee_denominator = 0;

        let fees = Fees {
            trade_fee_numerator,
            trade_fee_denominator,
            owner_trade_fee_numerator,
            owner_trade_fee_denominator,
            owner_withdraw_fee_numerator,
            owner_withdraw_fee_denominator,
            host_fee_numerator,
            host_fee_denominator,
        };
        let source_amount = 100;
        let curve = ConstantProductCurve {
            ..Default::default()
        };
        let swap_curve = SwapCurve {
            curve_type: CurveType::ConstantProduct,
            calculator: Arc::new(curve),
        };
        let result = swap_curve
            .swap(
                source_amount,
                swap_source_amount,
                swap_destination_amount,
                TradeDirection::AtoB,
                &fees,
            )
            .unwrap();
        assert_eq!(result.new_swap_source_amount, 1100);
        assert_eq!(result.destination_amount_swapped, 4504);
        assert_eq!(result.new_swap_destination_amount, 45496);
        assert_eq!(result.trade_fee, 1);
        assert_eq!(result.owner_fee, 0);
    }

    #[test]
    fn constant_product_owner_fee() {
        // calculation on https://github.com/solana-labs/solana-program-library/issues/341
        let swap_source_amount = 1000;
        let swap_destination_amount = 50000;
        let trade_fee_numerator = 0;
        let trade_fee_denominator = 0;
        let owner_trade_fee_numerator = 1;
        let owner_trade_fee_denominator = 100;
        let owner_withdraw_fee_numerator = 0;
        let owner_withdraw_fee_denominator = 0;
        let host_fee_numerator = 0;
        let host_fee_denominator = 0;
        let fees = Fees {
            trade_fee_numerator,
            trade_fee_denominator,
            owner_trade_fee_numerator,
            owner_trade_fee_denominator,
            owner_withdraw_fee_numerator,
            owner_withdraw_fee_denominator,
            host_fee_numerator,
            host_fee_denominator,
        };
        let source_amount: u128 = 100;
        let curve = ConstantProductCurve {
            ..Default::default()
        };
        let swap_curve = SwapCurve {
            curve_type: CurveType::ConstantProduct,
            calculator: Arc::new(curve),
        };
        let result = swap_curve
            .swap(
                source_amount,
                swap_source_amount,
                swap_destination_amount,
                TradeDirection::AtoB,
                &fees,
            )
            .unwrap();
        assert_eq!(result.new_swap_source_amount, 1100);
        assert_eq!(result.destination_amount_swapped, 4504);
        assert_eq!(result.new_swap_destination_amount, 45496);
        assert_eq!(result.trade_fee, 0);
        assert_eq!(result.owner_fee, 1);
    }

    #[test]
    fn constant_product_no_fee() {
        let swap_source_amount: u128 = 1_000;
        let swap_destination_amount: u128 = 50_000;
        let source_amount: u128 = 100;
        let curve = ConstantProductCurve::default();
        let fees = Fees::default();
        let swap_curve = SwapCurve {
            curve_type: CurveType::ConstantProduct,
            calculator: Arc::new(curve),
        };
        let result = swap_curve
            .swap(
                source_amount,
                swap_source_amount,
                swap_destination_amount,
                TradeDirection::AtoB,
                &fees,
            )
            .unwrap();
        assert_eq!(result.new_swap_source_amount, 1100);
        assert_eq!(result.destination_amount_swapped, 4545);
        assert_eq!(result.new_swap_destination_amount, 45455);
    }

    fn one_sided_deposit_vs_swap(
        source_amount: u128,
        swap_source_amount: u128,
        swap_destination_amount: u128,
        pool_supply: u128,
        fees: Fees,
    ) -> (u128, u128) {
        let curve = ConstantProductCurve::default();
        let swap_curve = SwapCurve {
            curve_type: CurveType::ConstantProduct,
            calculator: Arc::new(curve),
        };
        // do the A to B swap
        let results = swap_curve
            .swap(
                source_amount,
                swap_source_amount,
                swap_destination_amount,
                TradeDirection::AtoB,
                &fees,
            )
            .unwrap();

        // deposit just A, get pool tokens
        let deposit_pool_tokens = swap_curve
            .deposit_single_token_type(
                results.source_amount_swapped,
                swap_source_amount,
                swap_destination_amount,
                pool_supply,
                TradeDirection::AtoB,
                &fees,
            )
            .unwrap();
        let withdraw_pool_tokens = swap_curve
            .withdraw_single_token_type_exact_out(
                results.destination_amount_swapped,
                swap_source_amount + results.source_amount_swapped,
                swap_destination_amount,
                pool_supply + deposit_pool_tokens,
                TradeDirection::BtoA,
                &fees,
            )
            .unwrap();
        (withdraw_pool_tokens, deposit_pool_tokens)
    }

    #[test]
    fn one_sided_equals_swap_with_fee_specific() {
        let pool_supply: u128 = 1_000_000;
        let swap_source_amount: u128 = 1_000_000;
        let swap_destination_amount: u128 = 50_000_000;
        let source_amount: u128 = 10_000;
        let fees = Fees {
            trade_fee_numerator: 25,
            trade_fee_denominator: 1_000,
            owner_trade_fee_numerator: 5,
            owner_trade_fee_denominator: 1_000,
            ..Fees::default()
        };
        let (withdraw_pool_tokens, deposit_pool_tokens) = one_sided_deposit_vs_swap(
            source_amount,
            swap_source_amount,
            swap_destination_amount,
            pool_supply,
            fees,
        );
        // these checks *must* always hold
        assert!(withdraw_pool_tokens >= deposit_pool_tokens);
        let epsilon = 2;
        assert!(withdraw_pool_tokens - deposit_pool_tokens <= epsilon);

        // these checks may change if the calc is updated
        assert_eq!(withdraw_pool_tokens, 4914);
        assert_eq!(deposit_pool_tokens, 4912);
    }

    proptest! {
        #[test]
        fn one_sided_equals_swap_with_fee(
            (swap_source_amount, source_amount) in total_and_intermediate(u64::MAX),
            swap_destination_amount in 1..u64::MAX,
            pool_supply in 1..u64::MAX,
        ) {
            let fees = Fees {
                trade_fee_numerator: 25,
                trade_fee_denominator: 1_000,
                owner_trade_fee_numerator: 5,
                owner_trade_fee_denominator: 1_000,
                ..Fees::default()
            };
            let (withdraw_pool_tokens, deposit_pool_tokens) = one_sided_deposit_vs_swap(
                pool_supply.into(),
                swap_source_amount.into(),
                swap_destination_amount.into(),
                source_amount.into(),
                fees
            );
            // the cost to withdraw B must always be higher than the amount gained through deposit
            assert!(withdraw_pool_tokens >= deposit_pool_tokens);
        }

        #[test]
        fn one_sided_equals_swap_with_withdrawal_fee(
            (swap_source_amount, source_amount) in total_and_intermediate(u64::MAX),
            swap_destination_amount in 1..u64::MAX,
            pool_supply in 1..u64::MAX,
        ) {
            let fees = Fees {
                trade_fee_numerator: 25,
                trade_fee_denominator: 1_000,
                owner_trade_fee_numerator: 5,
                owner_trade_fee_denominator: 1_000,
                owner_withdraw_fee_numerator: 1,
                owner_withdraw_fee_denominator: 1_000,
                ..Fees::default()
            };
            let (withdraw_pool_tokens, deposit_pool_tokens) = one_sided_deposit_vs_swap(
                pool_supply.into(),
                swap_source_amount.into(),
                swap_destination_amount.into(),
                source_amount.into(),
                fees
            );
            // the cost to withdraw B must always be higher than the amount gained through deposit
            assert!(withdraw_pool_tokens >= deposit_pool_tokens);
        }

        #[test]
        fn one_sided_equals_swap_without_fee(
            (swap_source_amount, source_amount) in total_and_intermediate(u64::MAX),
            swap_destination_amount in 1..u64::MAX,
            pool_supply in 1..u64::MAX,
        ) {
            let fees = Fees::default();
            let (withdraw_pool_tokens, deposit_pool_tokens) = one_sided_deposit_vs_swap(
                pool_supply.into(),
                swap_source_amount.into(),
                swap_destination_amount.into(),
                source_amount.into(),
                fees
            );
            let difference = if withdraw_pool_tokens >= deposit_pool_tokens {
                withdraw_pool_tokens - deposit_pool_tokens
            } else {
                deposit_pool_tokens - withdraw_pool_tokens
            };
            // Accurate to one part in 1,000,000 -- without fees, it can go either
            // way due to vast differences in the pool token and trading token
            // amounts.
            // For example, if there's only 1 pool token and 1 destination token,
            // but a source amount of 1,000,000,000, we can lose up to 1,000,000,000
            // in precision during an operation.
            // See the proptests in calculator.rs for more specific versions.
            let epsilon = std::cmp::max(1, withdraw_pool_tokens / 1_000_000);
            assert!(
                difference <= epsilon,
                "difference between {} and {} expected to be less than {}, actually {}",
                withdraw_pool_tokens,
                deposit_pool_tokens,
                epsilon,
                difference
            );
        }
    }
}
