//! Instruction types

#![allow(clippy::too_many_arguments)]

use anchor_lang::{
    prelude::{Rent, System},
    solana_program::{
        instruction::Instruction, program_error::ProgramError, pubkey::Pubkey, sysvar::SysvarId,
    },
    Id, InstructionData, ToAccountMetas,
};
#[cfg(feature = "fuzz")]
use arbitrary::Arbitrary;

use crate::{curve::fees::Fees, instructions::CurveUserParameters, InitialSupply};

/// Initialize instruction data
#[derive(Debug, PartialEq)]
pub struct Initialize {
    /// all swap fees
    pub fees: Fees,
    /// swap curve info for pool, including CurveType and anything
    /// else that may be required
    pub curve: CurveUserParameters,
}

/// Swap instruction data
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
#[derive(Clone, Debug, PartialEq)]
pub struct Swap {
    /// SOURCE amount to transfer, output to DESTINATION is based on the exchange rate
    pub amount_in: u64,
    /// Minimum amount of DESTINATION token to output, prevents excessive slippage
    pub minimum_amount_out: u64,
}

/// DepositAllTokenTypes instruction data
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
#[derive(Clone, Debug, PartialEq)]
pub struct DepositAllTokenTypes {
    /// Pool token amount to transfer. token_a and token_b amount are set by
    /// the current exchange rate and size of the pool
    pub pool_token_amount: u64,
    /// Maximum token A amount to deposit, prevents excessive slippage
    pub maximum_token_a_amount: u64,
    /// Maximum token B amount to deposit, prevents excessive slippage
    pub maximum_token_b_amount: u64,
}

/// WithdrawAllTokenTypes instruction data
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
#[derive(Clone, Debug, PartialEq)]
pub struct WithdrawAllTokenTypes {
    /// Amount of pool tokens to burn. User receives an output of token a
    /// and b based on the percentage of the pool tokens that are returned.
    pub pool_token_amount: u64,
    /// Minimum amount of token A to receive, prevents excessive slippage
    pub minimum_token_a_amount: u64,
    /// Minimum amount of token B to receive, prevents excessive slippage
    pub minimum_token_b_amount: u64,
}

/// Deposit one token type, exact amount in instruction data
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
#[derive(Clone, Debug, PartialEq)]
pub struct DepositSingleTokenTypeExactAmountIn {
    /// Token amount to deposit
    pub source_token_amount: u64,
    /// Pool token amount to receive in exchange. The amount is set by
    /// the current exchange rate and size of the pool
    pub minimum_pool_token_amount: u64,
}

/// WithdrawSingleTokenTypeExactAmountOut instruction data
#[cfg_attr(feature = "fuzz", derive(Arbitrary))]
#[derive(Clone, Debug, PartialEq)]
pub struct WithdrawSingleTokenTypeExactAmountOut {
    /// Amount of token A or B to receive
    pub destination_token_amount: u64,
    /// Maximum amount of pool tokens to burn. User receives an output of token A
    /// or B based on the percentage of the pool tokens that are returned.
    pub maximum_pool_token_amount: u64,
}

/// Creates an 'initialize' instruction.
pub fn initialize_pool(
    program_id: &Pubkey,
    admin_authority: &Pubkey,
    pool: &Pubkey,
    swap_curve: &Pubkey,
    token_a_mint: &Pubkey,
    token_b_mint: &Pubkey,
    token_a_vault: &Pubkey,
    token_b_vault: &Pubkey,
    pool_authority: &Pubkey,
    pool_token_mint: &Pubkey,
    pool_token_fees_vault: &Pubkey,
    admin_authority_token_a_ata: &Pubkey,
    admin_authority_token_b_ata: &Pubkey,
    admin_authority_pool_token_ata: &Pubkey,
    pool_token_program_id: &Pubkey,
    token_a_program_id: &Pubkey,
    token_b_program_id: &Pubkey,
    fees: Fees,
    initial_supply: InitialSupply,
    curve_parameters: CurveUserParameters,
) -> Result<Instruction, ProgramError> {
    let data = super::instruction::InitializePool {
        initial_supply_a: initial_supply.initial_supply_a,
        initial_supply_b: initial_supply.initial_supply_b,
        fees,
        curve_parameters,
    }
    .data();

    let accounts = super::accounts::InitializePool {
        admin_authority: *admin_authority,
        pool: *pool,
        swap_curve: *swap_curve,
        pool_authority: *pool_authority,
        token_a_mint: *token_a_mint,
        token_b_mint: *token_b_mint,
        token_a_vault: *token_a_vault,
        token_b_vault: *token_b_vault,
        pool_token_mint: *pool_token_mint,
        pool_token_fees_vault: *pool_token_fees_vault,
        admin_authority_token_a_ata: *admin_authority_token_a_ata,
        admin_authority_token_b_ata: *admin_authority_token_b_ata,
        admin_authority_pool_token_ata: *admin_authority_pool_token_ata,
        system_program: System::id(),
        rent: Rent::id(),
        pool_token_program: *pool_token_program_id,
        token_a_token_program: *token_a_program_id,
        token_b_token_program: *token_b_program_id,
    }
    .to_account_metas(None);

    Ok(Instruction {
        program_id: *program_id,
        accounts,
        data,
    })
}

/// Creates a 'deposit_all_token_types' instruction.
pub fn deposit_all_token_types(
    program_id: &Pubkey,
    token_a_program_id: &Pubkey,
    token_b_program_id: &Pubkey,
    pool_token_program_id: &Pubkey,
    swap_pubkey: &Pubkey,
    authority_pubkey: &Pubkey,
    user_transfer_authority_pubkey: &Pubkey,
    deposit_token_a_pubkey: &Pubkey,
    deposit_token_b_pubkey: &Pubkey,
    swap_token_a_pubkey: &Pubkey,
    swap_token_b_pubkey: &Pubkey,
    pool_mint_pubkey: &Pubkey,
    destination_pubkey: &Pubkey,
    token_a_mint_pubkey: &Pubkey,
    token_b_mint_pubkey: &Pubkey,
    swap_curve: &Pubkey,
    instruction: DepositAllTokenTypes,
) -> Result<Instruction, ProgramError> {
    let data = super::instruction::DepositAllTokenTypes {
        pool_token_amount: instruction.pool_token_amount,
        maximum_token_a_amount: instruction.maximum_token_a_amount,
        maximum_token_b_amount: instruction.maximum_token_b_amount,
    }
    .data();

    let accounts = super::accounts::DepositAllTokenTypes {
        signer: *user_transfer_authority_pubkey,
        pool: *swap_pubkey,
        swap_curve: *swap_curve,
        pool_authority: *authority_pubkey,
        token_a_mint: *token_a_mint_pubkey,
        token_b_mint: *token_b_mint_pubkey,
        token_a_vault: *swap_token_a_pubkey,
        token_b_vault: *swap_token_b_pubkey,
        pool_token_mint: *pool_mint_pubkey,
        token_a_user_ata: *deposit_token_a_pubkey,
        token_b_user_ata: *deposit_token_b_pubkey,
        pool_token_user_ata: *destination_pubkey,
        pool_token_program: *pool_token_program_id,
        token_a_token_program: *token_a_program_id,
        token_b_token_program: *token_b_program_id,
    }
    .to_account_metas(None);

    Ok(Instruction {
        program_id: *program_id,
        accounts,
        data,
    })
}

/// Creates a 'withdraw_all_token_types' instruction.
pub fn withdraw_all_token_types(
    program_id: &Pubkey,
    pool_token_program_id: &Pubkey,
    token_a_program_id: &Pubkey,
    token_b_program_id: &Pubkey,
    pool: &Pubkey,
    authority_pubkey: &Pubkey,
    user_transfer_authority_pubkey: &Pubkey,
    pool_mint_pubkey: &Pubkey,
    fee_account_pubkey: &Pubkey,
    source_pubkey: &Pubkey,
    swap_token_a_pubkey: &Pubkey,
    swap_token_b_pubkey: &Pubkey,
    destination_token_a_pubkey: &Pubkey,
    destination_token_b_pubkey: &Pubkey,
    token_a_mint_pubkey: &Pubkey,
    token_b_mint_pubkey: &Pubkey,
    swap_curve: &Pubkey,
    instruction: WithdrawAllTokenTypes,
) -> Result<Instruction, ProgramError> {
    let data = super::instruction::WithdrawAllTokenTypes {
        pool_token_amount: instruction.pool_token_amount,
        minimum_token_a_amount: instruction.minimum_token_a_amount,
        minimum_token_b_amount: instruction.minimum_token_b_amount,
    }
    .data();

    let accounts = super::accounts::WithdrawAllTokenTypes {
        signer: *user_transfer_authority_pubkey,
        pool: *pool,
        swap_curve: *swap_curve,
        pool_authority: *authority_pubkey,
        token_a_mint: *token_a_mint_pubkey,
        token_b_mint: *token_b_mint_pubkey,
        token_a_vault: *swap_token_a_pubkey,
        token_b_vault: *swap_token_b_pubkey,
        pool_token_mint: *pool_mint_pubkey,
        pool_token_fees_vault: *fee_account_pubkey,
        token_a_user_ata: *destination_token_a_pubkey,
        token_b_user_ata: *destination_token_b_pubkey,
        pool_token_user_ata: *source_pubkey,
        pool_token_program: *pool_token_program_id,
        token_a_token_program: *token_a_program_id,
        token_b_token_program: *token_b_program_id,
    }
    .to_account_metas(None);

    Ok(Instruction {
        program_id: *program_id,
        accounts,
        data,
    })
}

/// Creates a 'deposit_single_token_type_exact_amount_in' instruction.
pub fn deposit_single_token_type(
    program_id: &Pubkey,
    source_token_program_id: &Pubkey,
    pool_token_program_id: &Pubkey,
    pool: &Pubkey,
    authority_pubkey: &Pubkey,
    user_transfer_authority_pubkey: &Pubkey,
    source_token_pubkey: &Pubkey,
    swap_token_a_pubkey: &Pubkey,
    swap_token_b_pubkey: &Pubkey,
    pool_mint_pubkey: &Pubkey,
    destination_pubkey: &Pubkey,
    source_mint_pubkey: &Pubkey,
    swap_curve: &Pubkey,
    instruction: DepositSingleTokenTypeExactAmountIn,
) -> Result<Instruction, ProgramError> {
    let data = super::instruction::DepositSingleTokenType {
        source_token_amount: instruction.source_token_amount,
        minimum_pool_token_amount: instruction.minimum_pool_token_amount,
    }
    .data();

    let accounts = super::accounts::DepositSingleTokenType {
        signer: *user_transfer_authority_pubkey,
        pool: *pool,
        swap_curve: *swap_curve,
        pool_authority: *authority_pubkey,
        source_token_mint: *source_mint_pubkey,
        token_a_vault: *swap_token_a_pubkey,
        token_b_vault: *swap_token_b_pubkey,
        pool_token_mint: *pool_mint_pubkey,
        source_token_user_ata: *source_token_pubkey,
        pool_token_user_ata: *destination_pubkey,
        pool_token_program: *pool_token_program_id,
        source_token_program: *source_token_program_id,
    }
    .to_account_metas(None);

    Ok(Instruction {
        program_id: *program_id,
        accounts,
        data,
    })
}

/// Creates a 'withdraw_single_token_type_exact_amount_out' instruction.
pub fn withdraw_single_token_type_exact_amount_out(
    program_id: &Pubkey,
    pool_token_program_id: &Pubkey,
    destination_token_program_id: &Pubkey,
    pool: &Pubkey,
    authority_pubkey: &Pubkey,
    user_transfer_authority_pubkey: &Pubkey,
    pool_mint_pubkey: &Pubkey,
    fee_account_pubkey: &Pubkey,
    pool_token_source_pubkey: &Pubkey,
    swap_token_a_pubkey: &Pubkey,
    swap_token_b_pubkey: &Pubkey,
    destination_pubkey: &Pubkey,
    destination_mint_pubkey: &Pubkey,
    swap_curve: &Pubkey,
    instruction: WithdrawSingleTokenTypeExactAmountOut,
) -> Result<Instruction, ProgramError> {
    let data = super::instruction::WithdrawSingleTokenType {
        destination_token_amount: instruction.destination_token_amount,
        maximum_pool_token_amount: instruction.maximum_pool_token_amount,
    }
    .data();

    let accounts = super::accounts::WithdrawSingleTokenType {
        signer: *user_transfer_authority_pubkey,
        pool: *pool,
        swap_curve: *swap_curve,
        pool_authority: *authority_pubkey,
        destination_token_mint: *destination_mint_pubkey,
        token_a_vault: *swap_token_a_pubkey,
        token_b_vault: *swap_token_b_pubkey,
        pool_token_mint: *pool_mint_pubkey,
        pool_token_fees_vault: *fee_account_pubkey,
        destination_token_user_ata: *destination_pubkey,
        pool_token_user_ata: *pool_token_source_pubkey,
        pool_token_program: *pool_token_program_id,
        destination_token_program: *destination_token_program_id,
    }
    .to_account_metas(None);

    Ok(Instruction {
        program_id: *program_id,
        accounts,
        data,
    })
}

/// Creates a 'swap' instruction.
pub fn swap(
    program_id: &Pubkey,
    source_token_program_id: &Pubkey,
    destination_token_program_id: &Pubkey,
    pool_token_program_id: &Pubkey,
    swap_pubkey: &Pubkey,
    authority_pubkey: &Pubkey,
    user_transfer_authority_pubkey: &Pubkey,
    source_pubkey: &Pubkey,
    swap_source_pubkey: &Pubkey,
    swap_destination_pubkey: &Pubkey,
    destination_pubkey: &Pubkey,
    pool_mint_pubkey: &Pubkey,
    pool_fee_pubkey: &Pubkey,
    source_mint_pubkey: &Pubkey,
    destination_mint_pubkey: &Pubkey,
    swap_curve_pubkey: &Pubkey,
    host_fee_pubkey: Option<&Pubkey>,
    instruction: Swap,
) -> Result<Instruction, ProgramError> {
    let data = super::instruction::Swap {
        amount_in: instruction.amount_in,
        minimum_amount_out: instruction.minimum_amount_out,
    }
    .data();

    let accounts = super::accounts::Swap {
        signer: *user_transfer_authority_pubkey,
        pool: *swap_pubkey,
        swap_curve: *swap_curve_pubkey,
        pool_authority: *authority_pubkey,
        source_mint: *source_mint_pubkey,
        destination_mint: *destination_mint_pubkey,
        source_vault: *swap_source_pubkey,
        destination_vault: *swap_destination_pubkey,
        pool_token_mint: *pool_mint_pubkey,
        pool_token_fees_vault: *pool_fee_pubkey,
        source_user_ata: *source_pubkey,
        destination_user_ata: *destination_pubkey,
        pool_token_host_fees_account: host_fee_pubkey.copied(),
        pool_token_program: *pool_token_program_id,
        source_token_program: *source_token_program_id,
        destination_token_program: *destination_token_program_id,
    }
    .to_account_metas(None);

    Ok(Instruction {
        program_id: *program_id,
        accounts,
        data,
    })
}
