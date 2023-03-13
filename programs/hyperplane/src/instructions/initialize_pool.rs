use anchor_lang::{
    accounts::{interface::Interface, interface_account::InterfaceAccount},
    prelude::{
        borsh::{BorshDeserialize, BorshSerialize},
        *,
    },
};
use anchor_spl::token_interface::{Mint, TokenAccount, TokenInterface};

use crate::{
    constraints::SWAP_CONSTRAINTS,
    curve::{base::SwapCurve, fees::Fees},
    error::SwapError,
    state::{Curve, SwapPool},
    to_u64,
    utils::{pool_token, seeds, swap_token},
};

#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum CurveUserParameters {
    ConstantProduct,
    ConstantPrice { token_b_price: u64 },
    Offset { token_b_offset: u64 },
    Stable { amp: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct InitialSupply {
    pub initial_supply_a: u64,
    pub initial_supply_b: u64,
}

pub fn handler(
    ctx: Context<InitializePool>,
    curve_parameters: CurveUserParameters,
    fees: Fees,
    initial_supply: InitialSupply,
) -> Result<()> {
    let InitialSupply {
        initial_supply_a,
        initial_supply_b,
    } = initial_supply;

    let curve_parameters = curve_parameters.to_curve_params(
        ctx.accounts.token_a_mint.decimals,
        ctx.accounts.token_b_mint.decimals,
    );

    let swap_curve = SwapCurve::new_from_params(curve_parameters)?;

    msg!(
        "Initialize parameters: swap_curve={:?}, initial_supply_a={}, initial_supply_b={}",
        swap_curve,
        initial_supply_a,
        initial_supply_b
    );

    swap_curve
        .calculator
        .validate_supply(initial_supply_a, initial_supply_b)?;

    let swap_constraints = &SWAP_CONSTRAINTS;

    // todo - elliot - not really needed as fee account owned by program
    if let Some(swap_constraints) = swap_constraints {
        // let owner_key = swap_constraints
        //     .owner_key
        //     .parse::<Pubkey>()
        //     .map_err(|_| SwapError::InvalidOwner)?;
        // if ctx.accounts.pool_token_fees_vault.owner != owner_key {
        //     msg!(
        //         "pool_token_fees_vault owner must be {} but was {}",
        //         owner_key,
        //         ctx.accounts.pool_token_fees_vault.owner
        //     );
        //     return Err(SwapError::InvalidOwner.into());
        // }
        swap_constraints.validate_curve(&swap_curve)?;
        swap_constraints.validate_fees(&fees)?;
    }
    fees.validate()?;
    swap_curve.calculator.validate()?;

    let initial_amount = swap_curve.calculator.new_pool_supply();

    let pool = &mut ctx.accounts.pool.load_init()?;
    pool.is_initialized = true.into();
    pool.pool_authority_bump_seed =
        u64::try_from(*ctx.bumps.get("pool_authority").unwrap()).unwrap();
    pool.pool_authority = ctx.accounts.pool_authority.key();
    pool.token_a_vault = ctx.accounts.token_a_vault.key();
    pool.token_b_vault = ctx.accounts.token_b_vault.key();
    pool.pool_token_mint = ctx.accounts.pool_token_mint.key();
    pool.token_a_mint = ctx.accounts.token_a_mint.key();
    pool.token_b_mint = ctx.accounts.token_b_mint.key();
    pool.pool_token_fees_vault = ctx.accounts.pool_token_fees_vault.key();
    pool.fees = fees;
    pool.curve_type = swap_curve.curve_type.into();
    pool.swap_curve = ctx.accounts.swap_curve.key();

    swap_token::transfer_from_user(
        ctx.accounts.token_a_token_program.to_account_info(),
        ctx.accounts.admin_authority_token_a_ata.to_account_info(),
        ctx.accounts.token_a_mint.to_account_info(),
        ctx.accounts.token_a_vault.to_account_info(),
        ctx.accounts.admin_authority.to_account_info(),
        initial_supply_a,
        ctx.accounts.token_a_mint.decimals,
    )?;
    swap_token::transfer_from_user(
        ctx.accounts.token_b_token_program.to_account_info(),
        ctx.accounts.admin_authority_token_b_ata.to_account_info(),
        ctx.accounts.token_b_mint.to_account_info(),
        ctx.accounts.token_b_vault.to_account_info(),
        ctx.accounts.admin_authority.to_account_info(),
        initial_supply_b,
        ctx.accounts.token_b_mint.decimals,
    )?;

    pool_token::mint(
        ctx.accounts.pool_token_program.to_account_info(),
        ctx.accounts.pool.to_account_info(),
        ctx.accounts.pool_token_mint.to_account_info(),
        ctx.accounts.pool_authority.to_account_info(),
        u64::try_from(*ctx.bumps.get("pool_authority").unwrap()).unwrap(),
        ctx.accounts
            .admin_authority_pool_token_ata
            .to_account_info(),
        to_u64!(initial_amount)?,
    )?;

    // Serialize the curve with a layout that is specific to the curve type
    swap_curve
        .calculator
        .try_dyn_serialize(ctx.accounts.swap_curve.try_borrow_mut_data()?)?;

    Ok(())
}

#[derive(Accounts)]
pub struct InitializePool<'info> {
    #[account(mut)]
    pub admin_authority: Signer<'info>,

    #[account(zero)]
    pub pool: AccountLoader<'info, SwapPool>,

    /// CHECK: This is checked in the handler -- TODO elliot - test checks better
    #[account(init,
        seeds = [seeds::SWAP_CURVE, pool.key().as_ref()],
        bump,
        payer = admin_authority,
        space = Curve::LEN,
    )]
    pub swap_curve: UncheckedAccount<'info>,

    /// CHECK: PDA owned by the program
    #[account(mut,
        seeds = [seeds::POOL_AUTHORITY, pool.key().as_ref()],
        bump
    )]
    pub pool_authority: AccountInfo<'info>,

    // todo - elliot - should we block if mint has freeze authority?
    // todo - elliot - token 2022 - should we block if mint has close authority?
    /// Token A mint
    // note - constraint repeated for clarity
    #[account(
        constraint = token_a_mint.key() != token_b_mint.key() @ SwapError::RepeatedMint,
        mint::token_program = token_a_token_program,
    )]
    pub token_a_mint: Box<InterfaceAccount<'info, Mint>>,

    // todo - elliot - should we block if mint has freeze authority?
    // todo - elliot - token 2022 - should we block if mint has close authority?
    /// Token B mint
    // note - constraint repeated for clarity
    #[account(
        constraint = token_a_mint.key() != token_b_mint.key() @ SwapError::RepeatedMint,
        mint::token_program = token_b_token_program,
    )]
    pub token_b_mint: Box<InterfaceAccount<'info, Mint>>,

    #[account(init,
        seeds = [seeds::TOKEN_A_VAULT, pool.key().as_ref(), token_a_mint.key().as_ref()],
        bump,
        payer = admin_authority,
        token::mint = token_a_mint,
        token::authority = pool_authority,
        token::token_program = token_a_token_program,
    )]
    pub token_a_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(init,
        seeds = [seeds::TOKEN_B_VAULT, pool.key().as_ref(), token_b_mint.key().as_ref()],
        bump,
        payer = admin_authority,
        token::mint = token_b_mint,
        token::authority = pool_authority,
        token::token_program = token_b_token_program,
    )]
    pub token_b_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    // todo - elliot - set no close authority, immutable? Should be default?
    #[account(init,
        seeds=[seeds::POOL_TOKEN_MINT, pool.key().as_ref()],
        bump,
        payer = admin_authority,
        mint::decimals = 6,
        mint::authority = pool_authority,
        mint::token_program = pool_token_program,
    )]
    pub pool_token_mint: Box<InterfaceAccount<'info, Mint>>,

    /// Token account to collect pool token fees into - designated to the pool admin authority
    #[account(init,
        seeds=[seeds::POOL_TOKEN_FEES_VAULT, pool.key().as_ref(), pool_token_mint.key().as_ref()],
        bump,
        payer = admin_authority,
        token::mint = pool_token_mint,
        token::authority = admin_authority,
        token::token_program = pool_token_program,
    )]
    pub pool_token_fees_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Admin authority's token A account to deposit initial liquidity from
    #[account(mut,
        token::mint = token_a_mint,
        token::authority = admin_authority,
        token::token_program = token_a_token_program,
    )]
    pub admin_authority_token_a_ata: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Admin authority's token B account to deposit initial liquidity from
    #[account(mut,
        token::mint = token_b_mint,
        token::authority = admin_authority,
        token::token_program = token_b_token_program,
    )]
    pub admin_authority_token_b_ata: Box<InterfaceAccount<'info, TokenAccount>>,

    /// Admin authority's pool token account to deposit the initially minted pool tokens into
    #[account(init,
        payer = admin_authority,
        token::mint = pool_token_mint,
        token::authority = admin_authority,
        token::token_program = pool_token_program,
    )]
    pub admin_authority_pool_token_ata: Box<InterfaceAccount<'info, TokenAccount>>,

    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
    /// The token program for the pool token mint
    pub pool_token_program: Interface<'info, TokenInterface>,
    /// The token program for the token A mint
    pub token_a_token_program: Interface<'info, TokenInterface>,
    /// The token program for the token B mint
    pub token_b_token_program: Interface<'info, TokenInterface>,
}

pub mod model {

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
    pub enum CurveParameters {
        ConstantProduct,
        ConstantPrice {
            token_b_price: u64,
        },
        Offset {
            token_b_offset: u64,
        },
        Stable {
            amp: u64,
            token_a_decimals: u8,
            token_b_decimals: u8,
        },
    }

    impl CurveUserParameters {
        pub fn to_curve_params(
            &self,
            token_a_decimals: u8,
            token_b_decimals: u8,
        ) -> CurveParameters {
            match self {
                CurveUserParameters::ConstantProduct => CurveParameters::ConstantProduct,
                CurveUserParameters::ConstantPrice { token_b_price } => {
                    CurveParameters::ConstantPrice {
                        token_b_price: *token_b_price,
                    }
                }
                CurveUserParameters::Offset { token_b_offset } => CurveParameters::Offset {
                    token_b_offset: *token_b_offset,
                },
                CurveUserParameters::Stable { amp } => CurveParameters::Stable {
                    amp: *amp,
                    token_a_decimals,
                    token_b_decimals,
                },
            }
        }
    }

    impl From<CurveParameters> for CurveUserParameters {
        fn from(curve_params: CurveParameters) -> Self {
            match curve_params {
                CurveParameters::ConstantProduct => CurveUserParameters::ConstantProduct,
                CurveParameters::ConstantPrice { token_b_price } => {
                    CurveUserParameters::ConstantPrice { token_b_price }
                }
                CurveParameters::Offset { token_b_offset } => {
                    CurveUserParameters::Offset { token_b_offset }
                }
                CurveParameters::Stable {
                    amp,
                    token_a_decimals: _,
                    token_b_decimals: _,
                } => CurveUserParameters::Stable { amp },
            }
        }
    }
}
