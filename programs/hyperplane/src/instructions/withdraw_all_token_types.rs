use anchor_lang::{
    accounts::{interface::Interface, interface_account::InterfaceAccount},
    prelude::*,
};
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};

use crate::{
    curve,
    curve::{base::SwapCurve, calculator::RoundDirection},
    emitted,
    error::SwapError,
    event, require_msg,
    state::{SwapPool, SwapState},
    to_u64, try_math,
    utils::{math::TryMath, pool_token, swap_token},
    withdraw_all_token_types::utils::validate_swap_inputs,
};

pub fn handler(
    ctx: Context<WithdrawAllTokenTypes>,
    pool_token_amount: u64,
    minimum_token_a_amount: u64,
    minimum_token_b_amount: u64,
) -> Result<event::WithdrawAllTokenTypes> {
    let pool = ctx.accounts.pool.load()?;
    validate_swap_inputs(&ctx, &pool)?;
    msg!(
        "Withdraw inputs: minimum_token_a_amount={}, minimum_token_b_amount={}, pool_token_amount={}",
        minimum_token_a_amount,
        minimum_token_b_amount,
        pool_token_amount,
    );
    let swap_curve = curve!(ctx.accounts.swap_curve, pool);

    let calculator = &swap_curve.calculator;

    msg!(
        "Swap pool inputs: swap_type={:?}, token_a_balance={}, token_b_balance={}, pool_token_supply={}",
        swap_curve.curve_type,
        ctx.accounts.token_a_vault.amount,
        ctx.accounts.token_b_vault.amount,
        ctx.accounts.pool_token_mint.supply,
    );

    let withdraw_fee = pool
        .fees()
        .owner_withdraw_fee(pool_token_amount.into())
        .map_err(|_| error!(SwapError::FeeCalculationFailure))?;
    let pool_token_amount = try_math!(u128::from(pool_token_amount).try_sub(withdraw_fee))?;

    msg!(
        "Withdrawal fee: fee={}, amount_after_fee={}",
        withdraw_fee,
        pool_token_amount
    );

    let results = calculator
        .pool_tokens_to_trading_tokens(
            pool_token_amount,
            u128::from(ctx.accounts.pool_token_mint.supply),
            u128::from(ctx.accounts.token_a_vault.amount),
            u128::from(ctx.accounts.token_b_vault.amount),
            RoundDirection::Floor,
        )
        .map_err(|_| error!(SwapError::ZeroTradingTokens))?;

    let token_a_amount = to_u64!(results.token_a_amount)?;
    let token_a_amount = std::cmp::min(ctx.accounts.token_a_vault.amount, token_a_amount);

    require_msg!(
        token_a_amount >= minimum_token_a_amount,
        SwapError::ExceededSlippage,
        &format!(
            "ExceededSlippage: token_a_amount={} < minimum_token_a_amount={}",
            token_a_amount, minimum_token_a_amount
        )
    );
    require!(
        token_a_amount > 0 || ctx.accounts.token_a_vault.amount == 0,
        SwapError::ZeroTradingTokens
    );

    let token_b_amount = to_u64!(results.token_b_amount)?;
    let token_b_amount = std::cmp::min(ctx.accounts.token_b_vault.amount, token_b_amount);

    require_msg!(
        token_b_amount >= minimum_token_b_amount,
        SwapError::ExceededSlippage,
        &format!(
            "ExceededSlippage: token_b_amount={} < minimum_token_b_amount={}",
            token_b_amount, minimum_token_b_amount
        )
    );
    require!(
        token_b_amount > 0 || ctx.accounts.token_b_vault.amount == 0,
        SwapError::ZeroTradingTokens
    );

    let withdraw_fee = to_u64!(withdraw_fee)?;
    if withdraw_fee > 0 {
        swap_token::transfer_from_user(
            ctx.accounts.pool_token_program.to_account_info(),
            ctx.accounts.pool_token_user_ata.to_account_info(),
            ctx.accounts.pool_token_mint.to_account_info(),
            ctx.accounts.pool_token_fees_vault.to_account_info(),
            ctx.accounts.signer.to_account_info(),
            withdraw_fee,
            ctx.accounts.pool_token_mint.decimals,
        )?;
    }

    msg!(
        "Withdraw outputs: token_a_to_receive={}, token_b_to_receive={}, pool_tokens_to_burn={}",
        token_a_amount,
        token_b_amount,
        pool_token_amount,
    );

    let pool_token_amount = to_u64!(pool_token_amount)?;
    pool_token::burn(
        ctx.accounts.pool_token_mint.to_account_info(),
        ctx.accounts.pool_token_user_ata.to_account_info(),
        ctx.accounts.signer.to_account_info(),
        ctx.accounts.pool_token_program.to_account_info(),
        pool_token_amount,
    )?;

    if token_a_amount > 0 {
        swap_token::transfer_from_vault(
            ctx.accounts.token_a_token_program.to_account_info(),
            ctx.accounts.pool.to_account_info(),
            ctx.accounts.token_a_vault.to_account_info(),
            ctx.accounts.token_a_mint.to_account_info(),
            ctx.accounts.token_a_user_ata.to_account_info(),
            ctx.accounts.pool_authority.to_account_info(),
            pool.pool_authority_bump_seed,
            token_a_amount,
            ctx.accounts.token_a_mint.decimals,
        )?;
    }
    if token_b_amount > 0 {
        swap_token::transfer_from_vault(
            ctx.accounts.token_b_token_program.to_account_info(),
            ctx.accounts.pool.to_account_info(),
            ctx.accounts.token_b_vault.to_account_info(),
            ctx.accounts.token_b_mint.to_account_info(),
            ctx.accounts.token_b_user_ata.to_account_info(),
            ctx.accounts.pool_authority.to_account_info(),
            pool.pool_authority_bump_seed,
            token_b_amount,
            ctx.accounts.token_b_mint.decimals,
        )?;
    }

    emitted!(event::WithdrawAllTokenTypes {
        pool_token_amount,
        token_a_amount,
        token_b_amount,
        fee: withdraw_fee,
    });
}

#[derive(Accounts)]
pub struct WithdrawAllTokenTypes<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,

    #[account(mut,
        has_one = swap_curve,
        has_one = pool_authority @ SwapError::InvalidProgramAddress,
        has_one = token_a_mint,
        has_one = token_b_mint,
        has_one = token_a_vault @ SwapError::IncorrectSwapAccount,
        has_one = token_b_vault @ SwapError::IncorrectSwapAccount,
        has_one = pool_token_mint @ SwapError::IncorrectPoolMint,
        has_one = pool_token_fees_vault @ SwapError::IncorrectFeeAccount,
    )]
    pub pool: AccountLoader<'info, SwapPool>,

    /// CHECK: has_one constraint on the pool
    pub swap_curve: UncheckedAccount<'info>,

    /// CHECK: has_one constraint on the pool
    pub pool_authority: AccountInfo<'info>,

    /// CHECK: has_one constraint on the pool
    pub token_a_mint: Box<InterfaceAccount<'info, Mint>>,

    /// CHECK: has_one constraint on the pool
    pub token_b_mint: Box<InterfaceAccount<'info, Mint>>,

    /// CHECK: has_one constraint on the pool
    #[account(mut)]
    pub token_a_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    /// CHECK: has_one constraint on the pool
    #[account(mut)]
    pub token_b_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    /// CHECK: has_one constraint on the pool
    #[account(mut)]
    pub pool_token_mint: Box<InterfaceAccount<'info, Mint>>,

    /// Account to collect fees into
    /// CHECK: has_one constraint on the pool
    #[account(mut)]
    pub pool_token_fees_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Signer's token A token account
    #[account(mut,
        token::mint = token_a_mint,
        token::token_program = token_a_token_program,
    )]
    pub token_a_user_ata: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Signer's token B token account
    #[account(mut,
        token::mint = token_b_mint,
        token::authority = token_a_user_ata.owner,
        token::token_program = token_b_token_program,
    )]
    pub token_b_user_ata: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Signer's pool token account
    #[account(mut,
        token::mint = pool_token_mint,
        token::authority = token_b_user_ata.owner,
        token::token_program = pool_token_program,
    )]
    pub pool_token_user_ata: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Token program for the pool token mint
    pub pool_token_program: Interface<'info, TokenInterface>,
    /// Token program for the source mint
    pub token_a_token_program: Interface<'info, TokenInterface>,
    /// Token program for the destination mint
    pub token_b_token_program: Interface<'info, TokenInterface>,
}

mod utils {
    use std::cell::Ref;

    use super::*;

    pub fn validate_swap_inputs(
        ctx: &Context<WithdrawAllTokenTypes>,
        pool: &Ref<SwapPool>,
    ) -> Result<()> {
        require_msg!(
            pool.token_a_vault != ctx.accounts.token_a_user_ata.key(),
            SwapError::IncorrectSwapAccount,
            &format!(
                "IncorrectSwapAccount: token_a_user_ata.key ({}) == token_a_vault.key ({})",
                ctx.accounts.token_a_user_ata.key(),
                pool.token_a_vault.key()
            )
        );
        require_msg!(
            pool.token_b_vault != ctx.accounts.token_b_user_ata.key(),
            SwapError::IncorrectSwapAccount,
            &format!(
                "IncorrectSwapAccount: token_b_user_ata.key ({}) == token_b_vault.key ({})",
                ctx.accounts.token_b_user_ata.key(),
                pool.token_b_vault.key()
            )
        );
        Ok(())
    }
}
