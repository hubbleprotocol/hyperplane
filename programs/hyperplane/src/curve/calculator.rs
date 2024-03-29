//! Swap calculations

use std::fmt::Debug;

use anchor_lang::Result;
#[cfg(feature = "fuzz")]
use arbitrary::Arbitrary;
use spl_math::precise_number::PreciseNumber;

use crate::{error::SwapError, require_msg};

/// Initial amount of pool tokens for swap contract, hard-coded to something
/// "sensible" given a maximum of u128.
/// Note that on Ethereum, Uniswap uses the geometric mean of all provided
/// input amounts, and Balancer uses 100 * 10 ^ 18.
pub const INITIAL_SWAP_POOL_AMOUNT: u128 = 1_000_000_000;

/// Hardcode the number of token types in a pool, used to calculate the
/// equivalent pool tokens for the owner trading fee.
pub const TOKENS_IN_POOL: u128 = 2;

/// The direction of a trade, since curves can be specialized to treat each
/// token differently (by adding offsets or weights)
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TradeDirection {
    /// Input token A, output token B
    AtoB,
    /// Input token B, output token A
    BtoA,
}

/// Utility to represent either token A or token B
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AorB {
    A,
    B,
}

/// The direction to round.  Used for pool token to trading token conversions to
/// avoid losing value on any deposit or withdrawal.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoundDirection {
    /// Floor the value, ie. 1.9 => 1, 1.1 => 1, 1.5 => 1
    Floor,
    /// Ceiling the value, ie. 1.9 => 2, 1.1 => 2, 1.5 => 2
    Ceiling,
}

impl TradeDirection {
    /// Given a trade direction, gives the opposite direction of the trade, so
    /// A to B becomes B to A, and vice versa
    pub fn opposite(&self) -> TradeDirection {
        match self {
            TradeDirection::AtoB => TradeDirection::BtoA,
            TradeDirection::BtoA => TradeDirection::AtoB,
        }
    }
}

/// Encodes all results of swapping from a source token to a destination token
#[derive(Debug, PartialEq)]
pub struct SwapWithoutFeesResult {
    /// Amount of source token swapped
    pub source_amount_swapped: u128,
    /// Amount of destination token swapped
    pub destination_amount_swapped: u128,
}

/// Encodes results of depositing both sides at once
#[derive(Debug, PartialEq)]
pub struct TradingTokenResult {
    /// Amount of token A
    pub token_a_amount: u128,
    /// Amount of token B
    pub token_b_amount: u128,
}

/// Trait for anchor serializing trait objects, required because structs that implement
/// `AccountSerialize` cannot be used as trait objects (as `dyn AccountSerialize`).
pub trait DynAccountSerialize {
    /// Only required function is to serialize given a trait object
    fn try_dyn_serialize(&self, dst: std::cell::RefMut<&mut [u8]>) -> Result<()>;
}

/// Trait representing operations required on a swap curve
pub trait CurveCalculator: Debug + DynAccountSerialize {
    /// Calculate how much destination token will be provided given an amount
    /// of source token.
    fn swap_without_fees(
        &self,
        source_amount: u128,
        pool_source_amount: u128,
        pool_destination_amount: u128,
        trade_direction: TradeDirection,
    ) -> Result<SwapWithoutFeesResult>;

    /// Get the supply for a new pool
    /// The default implementation is a Balancer-style fixed initial supply
    fn new_pool_supply(&self) -> u128 {
        INITIAL_SWAP_POOL_AMOUNT
    }

    /// Get the amount of trading tokens for the given amount of pool tokens,
    /// provided the total trading tokens and supply of pool tokens.
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
    ) -> Result<TradingTokenResult>;

    /// Validate that the given curve has no invalid parameters
    fn validate(&self) -> Result<()>;

    /// Validate the given supply on initialization. This is useful for curves
    /// that allow zero supply on one or both sides, since the standard constant
    /// product curve must have a non-zero supply on both sides.
    fn validate_supply(&self, token_a_amount: u64, token_b_amount: u64) -> Result<()> {
        require_msg!(
            token_a_amount > 0,
            SwapError::EmptySupply,
            "Token A supply must be greater than zero"
        );
        require_msg!(
            token_b_amount > 0,
            SwapError::EmptySupply,
            "Token B supply must be greater than zero"
        );
        Ok(())
    }

    /// Some curves function best and prevent attacks if we prevent deposits
    /// after initialization.  For example, the offset curve in `offset.rs`,
    /// which fakes supply on one side of the swap, allows the swap creator
    /// to steal value from all other depositors.
    fn allows_deposits(&self) -> bool {
        true
    }

    /// Calculates the total normalized value of the curve given the liquidity
    /// parameters.
    ///
    /// This value must have the dimension of `tokens ^ 1` For example, the
    /// standard Uniswap invariant has dimension `tokens ^ 2` since we are
    /// multiplying two token values together.  In order to normalize it, we
    /// also need to take the square root.
    ///
    /// This is useful for testing the curves, to make sure that value is not
    /// lost on any trade.  It can also be used to find out the relative value
    /// of pool tokens or liquidity tokens.
    fn normalized_value(
        &self,
        swap_token_a_amount: u128,
        swap_token_b_amount: u128,
    ) -> Result<PreciseNumber>;
}

/// Test helpers for curves
#[cfg(test)]
pub mod test {
    use proptest::prelude::*;
    use spl_math::uint::U256;

    use super::*;

    /// Test function checking that a swap never reduces the overall value of
    /// the pool.
    ///
    /// Since curve calculations use unsigned integers, there is potential for
    /// truncation at some point, meaning a potential for value to be lost in
    /// either direction if too much is given to the swapper.
    ///
    /// This test guarantees that the relative change in value will be at most
    /// 1 normalized token, and that the value will never decrease from a trade.
    pub fn check_curve_value_from_swap(
        curve: &dyn CurveCalculator,
        source_token_amount: u128,
        swap_source_amount: u128,
        swap_destination_amount: u128,
        trade_direction: TradeDirection,
    ) {
        let results = curve
            .swap_without_fees(
                source_token_amount,
                swap_source_amount,
                swap_destination_amount,
                trade_direction,
            )
            .unwrap();

        let (swap_token_a_amount, swap_token_b_amount) = match trade_direction {
            TradeDirection::AtoB => (swap_source_amount, swap_destination_amount),
            TradeDirection::BtoA => (swap_destination_amount, swap_source_amount),
        };
        let previous_value = curve
            .normalized_value(swap_token_a_amount, swap_token_b_amount)
            .unwrap();

        let new_swap_source_amount = swap_source_amount
            .checked_add(results.source_amount_swapped)
            .unwrap();
        let new_swap_destination_amount = swap_destination_amount
            .checked_sub(results.destination_amount_swapped)
            .unwrap();
        let (swap_token_a_amount, swap_token_b_amount) = match trade_direction {
            TradeDirection::AtoB => (new_swap_source_amount, new_swap_destination_amount),
            TradeDirection::BtoA => (new_swap_destination_amount, new_swap_source_amount),
        };

        let new_value = curve
            .normalized_value(swap_token_a_amount, swap_token_b_amount)
            .unwrap();
        assert!(new_value.greater_than_or_equal(&previous_value));

        let epsilon = 1; // Extremely close!
        let difference = new_value
            .checked_sub(&previous_value)
            .unwrap()
            .to_imprecise()
            .unwrap();
        assert!(difference <= epsilon);
    }

    /// Test function checking that a deposit never reduces the value of pool
    /// tokens.
    ///
    /// Since curve calculations use unsigned integers, there is potential for
    /// truncation at some point, meaning a potential for value to be lost if
    /// too much is given to the depositor.
    pub fn check_pool_value_from_deposit(
        curve: &dyn CurveCalculator,
        pool_token_amount: u128,
        pool_token_supply: u128,
        swap_token_a_amount: u128,
        swap_token_b_amount: u128,
    ) {
        let deposit_result = curve
            .pool_tokens_to_trading_tokens(
                pool_token_amount,
                pool_token_supply,
                swap_token_a_amount,
                swap_token_b_amount,
                RoundDirection::Ceiling,
            )
            .unwrap();
        let new_swap_token_a_amount = swap_token_a_amount + deposit_result.token_a_amount;
        let new_swap_token_b_amount = swap_token_b_amount + deposit_result.token_b_amount;
        let new_pool_token_supply = pool_token_supply + pool_token_amount;

        // the following inequality must hold:
        // new_token_a / new_pool_token_supply >= token_a / pool_token_supply
        // which reduces to:
        // new_token_a * pool_token_supply >= token_a * new_pool_token_supply

        // These numbers can be just slightly above u64 after the deposit, which
        // means that their multiplication can be just above the range of u128.
        // For ease of testing, we bump these up to U256.
        let pool_token_supply = U256::from(pool_token_supply);
        let new_pool_token_supply = U256::from(new_pool_token_supply);
        let swap_token_a_amount = U256::from(swap_token_a_amount);
        let new_swap_token_a_amount = U256::from(new_swap_token_a_amount);
        let swap_token_b_amount = U256::from(swap_token_b_amount);
        let new_swap_token_b_amount = U256::from(new_swap_token_b_amount);

        assert!(
            new_swap_token_a_amount * pool_token_supply
                >= swap_token_a_amount * new_pool_token_supply
        );
        assert!(
            new_swap_token_b_amount * pool_token_supply
                >= swap_token_b_amount * new_pool_token_supply
        );
    }

    /// Test function checking that a withdraw never reduces the value of pool
    /// tokens.
    ///
    /// Since curve calculations use unsigned integers, there is potential for
    /// truncation at some point, meaning a potential for value to be lost if
    /// too much is given to the depositor.
    pub fn check_pool_value_from_withdraw(
        curve: &dyn CurveCalculator,
        pool_token_amount: u128,
        pool_token_supply: u128,
        swap_token_a_amount: u128,
        swap_token_b_amount: u128,
    ) {
        let withdraw_result = curve
            .pool_tokens_to_trading_tokens(
                pool_token_amount,
                pool_token_supply,
                swap_token_a_amount,
                swap_token_b_amount,
                RoundDirection::Floor,
            )
            .unwrap();
        let new_swap_token_a_amount = swap_token_a_amount - withdraw_result.token_a_amount;
        let new_swap_token_b_amount = swap_token_b_amount - withdraw_result.token_b_amount;
        let new_pool_token_supply = pool_token_supply - pool_token_amount;

        let value = curve
            .normalized_value(swap_token_a_amount, swap_token_b_amount)
            .unwrap();
        // since we can get rounding issues on the pool value which make it seem that the
        // value per token has gone down, we bump it up by an epsilon of 1 to
        // cover all cases
        let new_value = curve
            .normalized_value(new_swap_token_a_amount, new_swap_token_b_amount)
            .unwrap();

        // the following inequality must hold:
        // new_pool_value / new_pool_token_supply >= pool_value / pool_token_supply
        // which can also be written:
        // new_pool_value * pool_token_supply >= pool_value * new_pool_token_supply

        let pool_token_supply = PreciseNumber::new(pool_token_supply).unwrap();
        let new_pool_token_supply = PreciseNumber::new(new_pool_token_supply).unwrap();
        assert!(new_value
            .checked_mul(&pool_token_supply)
            .unwrap()
            .greater_than_or_equal(&value.checked_mul(&new_pool_token_supply).unwrap()));
    }

    prop_compose! {
        pub fn total_and_intermediate(max_value: u64)(total in 1..max_value)
                        (intermediate in 1..total, total in Just(total))
                        -> (u64, u64) {
           (total, intermediate)
       }
    }
}
