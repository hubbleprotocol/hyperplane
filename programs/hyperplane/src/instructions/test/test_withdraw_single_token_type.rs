use anchor_lang::{error::ErrorCode as AnchorError, prelude::*};
use anchor_spl::{
    token::spl_token,
    token_2022::{
        spl_token_2022,
        spl_token_2022::{
            error::TokenError,
            extension::{transfer_fee::TransferFee, StateWithExtensions},
            state::{Account, Mint},
        },
    },
};
use solana_sdk::account::{Account as SolanaAccount, WritableAccount};
use test_case::test_case;

use crate::{
    curve::{base::SwapCurve, calculator::TradeDirection, fees::Fees},
    error::SwapError,
    instructions::test::runner::{
        processor::{do_process_instruction, SwapAccountInfo, SwapTransferFees},
        token,
    },
    ix,
    model::CurveParameters,
    utils::seeds,
    InitialSupply,
};

#[test_case(spl_token::id(), spl_token::id(), spl_token::id(); "all-token")]
#[test_case(spl_token::id(), spl_token_2022::id(), spl_token_2022::id(); "mixed-pool-token")]
#[test_case(spl_token_2022::id(), spl_token_2022::id(), spl_token_2022::id(); "all-token-2022")]
#[test_case(spl_token_2022::id(), spl_token_2022::id(), spl_token::id(); "a-only-token-2022")]
#[test_case(spl_token_2022::id(), spl_token::id(), spl_token_2022::id(); "b-only-token-2022")]
fn test_withdraw_one_exact_out(
    pool_token_program_id: Pubkey,
    token_a_program_id: Pubkey,
    token_b_program_id: Pubkey,
) {
    let user_key = Pubkey::new_unique();
    let trade_fee_numerator = 1;
    let trade_fee_denominator = 2;
    let owner_trade_fee_numerator = 1;
    let owner_trade_fee_denominator = 10;
    let owner_withdraw_fee_numerator = 1;
    let owner_withdraw_fee_denominator = 5;
    let host_fee_numerator = 7;
    let host_fee_denominator = 100;

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

    let token_a_amount = 100_000;
    let token_b_amount = 200_000;
    let curve_params = CurveParameters::ConstantProduct;
    let swap_curve = SwapCurve::new_from_params(curve_params.clone()).unwrap();

    let withdrawer_key = Pubkey::new_unique();
    let initial_a = token_a_amount / 10;
    let initial_b = token_b_amount / 10;
    let initial_pool = swap_curve.calculator.new_pool_supply() / 10;
    let maximum_pool_token_amount = u64::try_from(initial_pool / 4).unwrap();
    let destination_a_amount = initial_a / 40;
    let destination_b_amount = initial_b / 40;

    let mut accounts = SwapAccountInfo::new(
        &user_key,
        fees,
        SwapTransferFees::default(),
        curve_params,
        InitialSupply {
            initial_supply_a: token_a_amount,
            initial_supply_b: token_b_amount,
        },
        &pool_token_program_id,
        &token_a_program_id,
        &token_b_program_id,
    );

    // swap not initialized
    {
        let (token_a_key, mut token_a_account) = token::create_token_account(
            &accounts.token_a_program_id,
            &accounts.token_a_mint_key,
            &mut accounts.token_a_mint_account,
            &user_key,
            &withdrawer_key,
            initial_a,
        );
        // use token B mint because pool mint not initialized
        let (pool_key, mut pool_account) = token::create_token_account(
            &accounts.token_b_program_id,
            &accounts.token_b_mint_key,
            &mut accounts.token_b_mint_account,
            &user_key,
            &withdrawer_key,
            0,
        );
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::AccountDiscriminatorMismatch.into()
            )),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                destination_a_amount,
                maximum_pool_token_amount,
            )
        );
    }

    accounts.initialize_pool().unwrap();

    // wrong owner for swap account
    {
        let (
            token_a_key,
            mut token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &withdrawer_key, initial_a, initial_b, 0);
        let old_swap_account = accounts.pool_account;
        let mut wrong_swap_account = old_swap_account.clone();
        wrong_swap_account.owner = Pubkey::new_unique();
        accounts.pool_account = wrong_swap_account;
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::AccountOwnedByWrongProgram.into()
            )),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                destination_a_amount,
                maximum_pool_token_amount,
            )
        );
        accounts.pool_account = old_swap_account;
    }

    // wrong pool authority key
    {
        let (
            _token_a_key,
            _token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &withdrawer_key, initial_a, initial_b, 0);
        let old_authority = accounts.pool_authority;
        let (bad_authority_key, _bump_seed) = Pubkey::find_program_address(
            &[seeds::POOL_AUTHORITY, accounts.pool.as_ref()],
            &accounts.pool_token_program_id,
        );
        accounts.pool_authority = bad_authority_key;
        assert_eq!(
            Err(SwapError::InvalidProgramAddress.into()),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_b_key,
                &mut token_b_account,
                destination_b_amount,
                maximum_pool_token_amount,
            )
        );
        accounts.pool_authority = old_authority;
    }

    // not enough pool tokens
    {
        let (
            _token_a_key,
            _token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            maximum_pool_token_amount / 1000,
        );
        assert_eq!(
            Err(TokenError::InsufficientFunds.into()),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_b_key,
                &mut token_b_account,
                destination_b_amount,
                maximum_pool_token_amount,
            )
        );
    }

    // wrong pool token account
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            _pool_key,
            _pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            maximum_pool_token_amount,
            initial_b,
            maximum_pool_token_amount,
        );
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenMint.into()
            )),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                destination_b_amount,
                maximum_pool_token_amount,
            )
        );
    }

    // wrong pool fee account
    {
        let (
            token_a_key,
            mut token_a_account,
            _token_b_key,
            _token_b_account,
            wrong_pool_key,
            wrong_pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            maximum_pool_token_amount,
        );
        let (
            _token_a_key,
            _token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            maximum_pool_token_amount,
        );
        let old_pool_fee_account = accounts.pool_token_fees_vault_account;
        let old_pool_fee_key = accounts.pool_token_fees_vault_key;
        accounts.pool_token_fees_vault_account = wrong_pool_account;
        accounts.pool_token_fees_vault_key = wrong_pool_key;
        assert_eq!(
            Err(SwapError::IncorrectFeeAccount.into()),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                destination_a_amount,
                maximum_pool_token_amount,
            )
        );
        accounts.pool_token_fees_vault_account = old_pool_fee_account;
        accounts.pool_token_fees_vault_key = old_pool_fee_key;
    }

    // no approval
    {
        let (
            token_a_key,
            mut token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            0,
            0,
            maximum_pool_token_amount,
        );
        let user_transfer_authority_key = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);

        assert_eq!(
            Err(TokenError::OwnerMismatch.into()),
            do_process_instruction(
                ix::withdraw_single_token_type_exact_amount_out(
                    &crate::id(),
                    &accounts.pool_token_program_id,
                    &token_a_program_id,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &user_transfer_authority_key,
                    &accounts.pool_token_mint_key,
                    &accounts.pool_token_fees_vault_key,
                    &pool_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &token_a_key,
                    &accounts.token_a_mint_key,
                    &accounts.swap_curve_key,
                    ix::WithdrawSingleTokenTypeExactAmountOut {
                        destination_token_amount: destination_a_amount,
                        maximum_pool_token_amount,
                    }
                )
                .unwrap(),
                vec![
                    &mut SolanaAccount::default(),
                    &mut accounts.pool_account,
                    &mut accounts.swap_curve_account,
                    &mut SolanaAccount::default(),
                    &mut accounts.token_a_mint_account,
                    &mut accounts.token_a_vault_account,
                    &mut accounts.token_b_vault_account,
                    &mut accounts.pool_token_mint_account,
                    &mut accounts.pool_token_fees_vault_account,
                    &mut token_a_account,
                    &mut pool_account,
                    &mut exe.clone(), // pool_token_program
                    &mut exe.clone(), // destination_token_program
                ],
            )
        );
    }

    // wrong destination token program id
    {
        let (
            token_a_key,
            mut token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            maximum_pool_token_amount,
        );
        let wrong_key = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);

        assert_eq!(
            Err(ProgramError::Custom(AnchorError::InvalidProgramId.into())),
            do_process_instruction(
                ix::withdraw_single_token_type_exact_amount_out(
                    &crate::id(),
                    &accounts.pool_token_program_id,
                    &wrong_key,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &withdrawer_key,
                    &accounts.pool_token_mint_key,
                    &accounts.pool_token_fees_vault_key,
                    &pool_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &token_a_key,
                    &accounts.token_a_mint_key,
                    &accounts.swap_curve_key,
                    ix::WithdrawSingleTokenTypeExactAmountOut {
                        destination_token_amount: destination_a_amount,
                        maximum_pool_token_amount,
                    }
                )
                .unwrap(),
                vec![
                    &mut SolanaAccount::default(),
                    &mut accounts.pool_account,
                    &mut accounts.swap_curve_account,
                    &mut SolanaAccount::default(),
                    &mut accounts.token_a_mint_account,
                    &mut accounts.token_a_vault_account,
                    &mut accounts.token_b_vault_account,
                    &mut accounts.pool_token_mint_account,
                    &mut accounts.pool_token_fees_vault_account,
                    &mut token_a_account,
                    &mut pool_account,
                    &mut exe.clone(),
                    &mut exe.clone(),
                ],
            )
        );
    }

    // wrong pool token program id
    {
        let (
            token_a_key,
            mut token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            maximum_pool_token_amount,
        );
        let wrong_key = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);

        assert_eq!(
            Err(ProgramError::Custom(AnchorError::InvalidProgramId.into())),
            do_process_instruction(
                ix::withdraw_single_token_type_exact_amount_out(
                    &crate::id(),
                    &wrong_key,
                    &accounts.token_a_program_id,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &accounts.pool_authority,
                    &accounts.pool_token_mint_key,
                    &accounts.pool_token_fees_vault_key,
                    &pool_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &token_a_key,
                    &accounts.token_a_mint_key,
                    &accounts.swap_curve_key,
                    ix::WithdrawSingleTokenTypeExactAmountOut {
                        destination_token_amount: destination_a_amount,
                        maximum_pool_token_amount,
                    }
                )
                .unwrap(),
                vec![
                    &mut SolanaAccount::default(),
                    &mut accounts.pool_account,
                    &mut accounts.swap_curve_account,
                    &mut SolanaAccount::default(),
                    &mut accounts.token_a_mint_account,
                    &mut accounts.token_a_vault_account,
                    &mut accounts.token_b_vault_account,
                    &mut accounts.pool_token_mint_account,
                    &mut accounts.pool_token_fees_vault_account,
                    &mut token_a_account,
                    &mut pool_account,
                    &mut exe.clone(),
                    &mut exe.clone(),
                ],
            )
        );
    }

    // wrong swap token accounts
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            initial_pool.try_into().unwrap(),
        );

        let old_a_key = accounts.token_a_vault_key;
        let old_a_account = accounts.token_a_vault_account;

        accounts.token_a_vault_key = token_a_key;
        accounts.token_a_vault_account = token_a_account.clone();

        // wrong swap token a account
        assert_eq!(
            Err(SwapError::IncorrectSwapAccount.into()),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                destination_a_amount,
                maximum_pool_token_amount,
            )
        );

        accounts.token_a_vault_key = old_a_key;
        accounts.token_a_vault_account = old_a_account;

        let old_b_key = accounts.token_b_vault_key;
        let old_b_account = accounts.token_b_vault_account;

        accounts.token_b_vault_key = token_b_key;
        accounts.token_b_vault_account = token_b_account.clone();

        // wrong swap token b account
        assert_eq!(
            Err(SwapError::IncorrectSwapAccount.into()),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_b_key,
                &mut token_b_account,
                destination_b_amount,
                maximum_pool_token_amount,
            )
        );

        accounts.token_b_vault_key = old_b_key;
        accounts.token_b_vault_account = old_b_account;
    }

    // wrong mint
    {
        let (
            token_a_key,
            mut token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            initial_pool.try_into().unwrap(),
        );
        let (pool_mint_key, pool_mint_account) = token::create_mint(
            &accounts.pool_token_program_id,
            &accounts.pool_authority,
            None,
            None,
            &TransferFee::default(),
        );
        let old_pool_key = accounts.pool_token_mint_key;
        let old_pool_account = accounts.pool_token_mint_account;
        accounts.pool_token_mint_key = pool_mint_key;
        accounts.pool_token_mint_account = pool_mint_account;

        assert_eq!(
            Err(SwapError::IncorrectPoolMint.into()),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                destination_a_amount,
                maximum_pool_token_amount,
            )
        );

        accounts.pool_token_mint_key = old_pool_key;
        accounts.pool_token_mint_account = old_pool_account;
    }

    // slippage exceeded
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            maximum_pool_token_amount,
        );

        // maximum pool token amount too low
        assert_eq!(
            Err(SwapError::ExceededSlippage.into()),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                destination_a_amount,
                maximum_pool_token_amount / 1000,
            )
        );
        assert_eq!(
            Err(SwapError::ExceededSlippage.into()),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_b_key,
                &mut token_b_account,
                destination_b_amount,
                maximum_pool_token_amount / 1000,
            )
        );
    }

    // invalid input: can't use swap pool tokens as destination
    {
        let (
            _token_a_key,
            _token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            maximum_pool_token_amount,
        );
        let swap_token_a_key = accounts.token_a_vault_key;
        let mut swap_token_a_account = accounts.get_token_account(&swap_token_a_key).clone();
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenOwner.into()
            )),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &swap_token_a_key,
                &mut swap_token_a_account,
                destination_a_amount,
                maximum_pool_token_amount,
            )
        );
        let swap_token_b_key = accounts.token_b_vault_key;
        let mut swap_token_b_account = accounts.get_token_account(&swap_token_b_key).clone();
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenOwner.into()
            )),
            accounts.withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &swap_token_b_key,
                &mut swap_token_b_account,
                destination_b_amount,
                maximum_pool_token_amount,
            )
        );
    }

    // correct withdrawal
    {
        let (
            token_a_key,
            mut token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            initial_pool.try_into().unwrap(),
        );

        let swap_token_a =
            StateWithExtensions::<Account>::unpack(&accounts.token_a_vault_account.data).unwrap();
        let swap_token_b =
            StateWithExtensions::<Account>::unpack(&accounts.token_b_vault_account.data).unwrap();
        let pool_mint =
            StateWithExtensions::<Mint>::unpack(&accounts.pool_token_mint_account.data).unwrap();

        let pool_token_amount = accounts
            .swap_curve
            .withdraw_single_token_type_exact_out(
                destination_a_amount.try_into().unwrap(),
                swap_token_a.base.amount.try_into().unwrap(),
                swap_token_b.base.amount.try_into().unwrap(),
                pool_mint.base.supply.try_into().unwrap(),
                TradeDirection::AtoB,
                &accounts.fees,
            )
            .unwrap();
        let withdraw_fee = accounts.fees.owner_withdraw_fee(pool_token_amount).unwrap();

        accounts
            .withdraw_single_token_type_exact_amount_out(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                destination_a_amount,
                maximum_pool_token_amount,
            )
            .unwrap();

        let swap_token_a =
            StateWithExtensions::<Account>::unpack(&accounts.token_a_vault_account.data).unwrap();

        assert_eq!(
            swap_token_a.base.amount,
            token_a_amount - destination_a_amount
        );
        let token_a = StateWithExtensions::<Account>::unpack(&token_a_account.data).unwrap();
        assert_eq!(token_a.base.amount, initial_a + destination_a_amount);

        let pool_account = StateWithExtensions::<Account>::unpack(&pool_account.data).unwrap();
        assert_eq!(
            pool_account.base.amount,
            u64::try_from(initial_pool - pool_token_amount - withdraw_fee).unwrap()
        );
        let fee_account =
            StateWithExtensions::<Account>::unpack(&accounts.pool_token_fees_vault_account.data)
                .unwrap();
        assert_eq!(
            fee_account.base.amount,
            u64::try_from(withdraw_fee).unwrap()
        );
    }
}
