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
    curve::{base::SwapCurve, calculator::RoundDirection, fees::Fees},
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
fn test_withdraw(
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

    let token_a_amount = 1000;
    let token_b_amount = 2000;
    let curve_params = CurveParameters::ConstantProduct;
    let swap_curve = SwapCurve::new_from_params(curve_params.clone()).unwrap();

    let withdrawer_key = Pubkey::new_unique();
    let initial_a = token_a_amount / 10;
    let initial_b = token_b_amount / 10;
    let initial_pool = swap_curve.calculator.new_pool_supply() / 10;
    let withdraw_amount = initial_pool / 4;
    let minimum_token_a_amount = initial_a / 40;
    let minimum_token_b_amount = initial_b / 40;

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
        let (token_b_key, mut token_b_account) = token::create_token_account(
            &accounts.token_b_program_id,
            &accounts.token_b_mint_key,
            &mut accounts.token_b_mint_account,
            &user_key,
            &withdrawer_key,
            initial_b,
        );
        // use token A mint because pool mint not initialized
        let (pool_key, mut pool_account) = token::create_token_account(
            &accounts.token_a_program_id,
            &accounts.token_a_mint_key,
            &mut accounts.token_a_mint_account,
            &user_key,
            &withdrawer_key,
            0,
        );
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::AccountDiscriminatorMismatch.into()
            )),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            )
        );
    }

    accounts.initialize_pool().unwrap();

    // wrong owner for swap account
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
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
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            )
        );
        accounts.pool_account = old_swap_account;
    }

    // wrong pool authority
    {
        let (
            token_a_key,
            mut token_a_account,
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
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            )
        );
        accounts.pool_authority = old_authority;
    }

    // not enough pool tokens
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
            u64::try_from(withdraw_amount).unwrap() / 2u64,
        );
        assert_eq!(
            Err(TokenError::InsufficientFunds.into()),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount / 2,
                minimum_token_b_amount / 2,
            )
        );
    }

    // wrong token a / b accounts
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
            withdraw_amount.try_into().unwrap(),
        );
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenMint.into()
            )),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_b_key,
                &mut token_b_account,
                &token_a_key,
                &mut token_a_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
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
            initial_a,
            initial_b,
            withdraw_amount.try_into().unwrap(),
        );
        let (
            wrong_token_a_key,
            mut wrong_token_a_account,
            _token_b_key,
            _token_b_account,
            _pool_key,
            _pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            withdraw_amount.try_into().unwrap(),
            initial_b,
            withdraw_amount.try_into().unwrap(),
        );
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenMint.into()
            )),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &wrong_token_a_key,
                &mut wrong_token_a_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            )
        );
    }

    // wrong pool fee account
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            wrong_pool_key,
            wrong_pool_account,
        ) = accounts.setup_token_accounts(
            &user_key,
            &withdrawer_key,
            initial_a,
            initial_b,
            withdraw_amount.try_into().unwrap(),
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
            withdraw_amount.try_into().unwrap(),
        );
        let old_pool_fee_account = accounts.pool_token_fees_vault_account;
        let old_pool_fee_key = accounts.pool_token_fees_vault_key;
        accounts.pool_token_fees_vault_account = wrong_pool_account;
        accounts.pool_token_fees_vault_key = wrong_pool_key;
        assert_eq!(
            Err(SwapError::IncorrectFeeAccount.into()),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            ),
        );
        accounts.pool_token_fees_vault_account = old_pool_fee_account;
        accounts.pool_token_fees_vault_key = old_pool_fee_key;
    }

    // no approval
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
            0,
            0,
            withdraw_amount.try_into().unwrap(),
        );
        let user_transfer_authority_key = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);

        assert_eq!(
            Err(TokenError::OwnerMismatch.into()),
            do_process_instruction(
                ix::withdraw_all_token_types(
                    &crate::id(),
                    &accounts.pool_token_program_id,
                    &token_a_program_id,
                    &token_b_program_id,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &user_transfer_authority_key,
                    &accounts.pool_token_mint_key,
                    &accounts.pool_token_fees_vault_key,
                    &pool_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &token_a_key,
                    &token_b_key,
                    &accounts.token_a_mint_key,
                    &accounts.token_b_mint_key,
                    &accounts.swap_curve_key,
                    ix::WithdrawAllTokenTypes {
                        pool_token_amount: withdraw_amount.try_into().unwrap(),
                        minimum_token_a_amount,
                        minimum_token_b_amount,
                    }
                )
                .unwrap(),
                vec![
                    &mut SolanaAccount::default(),
                    &mut accounts.pool_account,
                    &mut accounts.swap_curve_account,
                    &mut SolanaAccount::default(),
                    &mut accounts.token_a_mint_account,
                    &mut accounts.token_b_mint_account,
                    &mut accounts.token_a_vault_account,
                    &mut accounts.token_b_vault_account,
                    &mut accounts.pool_token_mint_account,
                    &mut accounts.pool_token_fees_vault_account,
                    &mut token_a_account,
                    &mut token_b_account,
                    &mut pool_account,
                    &mut exe.clone(), // pool_token_program
                    &mut exe.clone(), // token_a_token_program
                    &mut exe.clone(), // token_b_token_program
                ],
            )
        );
    }

    // wrong pool token program id
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
            withdraw_amount.try_into().unwrap(),
        );
        let wrong_key = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);

        assert_eq!(
            Err(ProgramError::Custom(AnchorError::InvalidProgramId.into())),
            do_process_instruction(
                ix::withdraw_all_token_types(
                    &crate::id(),
                    &wrong_key,
                    &accounts.token_a_program_id,
                    &accounts.token_b_program_id,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &accounts.pool_authority,
                    &accounts.pool_token_mint_key,
                    &accounts.pool_token_fees_vault_key,
                    &pool_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &token_a_key,
                    &token_b_key,
                    &accounts.token_a_mint_key,
                    &accounts.token_b_mint_key,
                    &accounts.swap_curve_key,
                    ix::WithdrawAllTokenTypes {
                        pool_token_amount: withdraw_amount.try_into().unwrap(),
                        minimum_token_a_amount,
                        minimum_token_b_amount,
                    },
                )
                .unwrap(),
                vec![
                    &mut SolanaAccount::default(),
                    &mut accounts.pool_account,
                    &mut accounts.swap_curve_account,
                    &mut SolanaAccount::default(),
                    &mut accounts.token_a_mint_account,
                    &mut accounts.token_b_mint_account,
                    &mut accounts.token_a_vault_account,
                    &mut accounts.token_b_vault_account,
                    &mut accounts.pool_token_mint_account,
                    &mut accounts.pool_token_fees_vault_account,
                    &mut token_a_account,
                    &mut token_b_account,
                    &mut pool_account,
                    &mut exe.clone(), // pool_token_program
                    &mut exe.clone(), // token_a_token_program
                    &mut exe.clone(), // token_b_token_program
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
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
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
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
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
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            )
        );

        accounts.pool_token_mint_key = old_pool_key;
        accounts.pool_token_mint_account = old_pool_account;
    }

    // withdrawing 1 pool token fails because it equates to 0 output tokens
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
        assert_eq!(
            Err(SwapError::ZeroTradingTokens.into()),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                1,
                0,
                0,
            )
        );
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
            initial_pool.try_into().unwrap(),
        );
        // minimum A amount out too high
        assert_eq!(
            Err(SwapError::ExceededSlippage.into()),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount * 10,
                minimum_token_b_amount,
            )
        );
        // minimum B amount out too high
        assert_eq!(
            Err(SwapError::ExceededSlippage.into()),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount * 10,
            )
        );
    }

    // invalid input: can't use swap pool tokens as destination
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
        let swap_token_a_key = accounts.token_a_vault_key;
        let mut swap_token_a_account = accounts.get_token_account(&swap_token_a_key).clone();
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenOwner.into()
            )),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &swap_token_a_key,
                &mut swap_token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            )
        );
        let swap_token_b_key = accounts.token_b_vault_key;
        let mut swap_token_b_account = accounts.get_token_account(&swap_token_b_key).clone();
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenOwner.into()
            )),
            accounts.withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &swap_token_b_key,
                &mut swap_token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            )
        );
    }

    // correct withdrawal
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

        accounts
            .withdraw_all_token_types(
                &withdrawer_key,
                &pool_key,
                &mut pool_account,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                withdraw_amount.try_into().unwrap(),
                minimum_token_a_amount,
                minimum_token_b_amount,
            )
            .unwrap();

        let swap_token_a =
            StateWithExtensions::<Account>::unpack(&accounts.token_a_vault_account.data).unwrap();
        let swap_token_b =
            StateWithExtensions::<Account>::unpack(&accounts.token_b_vault_account.data).unwrap();
        let pool_mint =
            StateWithExtensions::<Mint>::unpack(&accounts.pool_token_mint_account.data).unwrap();
        let withdraw_fee = accounts.fees.owner_withdraw_fee(withdraw_amount).unwrap();
        let results = accounts
            .swap_curve
            .calculator
            .pool_tokens_to_trading_tokens(
                withdraw_amount - withdraw_fee,
                pool_mint.base.supply.try_into().unwrap(),
                swap_token_a.base.amount.try_into().unwrap(),
                swap_token_b.base.amount.try_into().unwrap(),
                RoundDirection::Floor,
            )
            .unwrap();
        assert_eq!(
            swap_token_a.base.amount,
            token_a_amount - u64::try_from(results.token_a_amount).unwrap()
        );
        assert_eq!(
            swap_token_b.base.amount,
            token_b_amount - u64::try_from(results.token_b_amount).unwrap()
        );
        let token_a = StateWithExtensions::<Account>::unpack(&token_a_account.data).unwrap();
        assert_eq!(
            token_a.base.amount,
            initial_a + u64::try_from(results.token_a_amount).unwrap()
        );
        let token_b = StateWithExtensions::<Account>::unpack(&token_b_account.data).unwrap();
        assert_eq!(
            token_b.base.amount,
            initial_b + u64::try_from(results.token_b_amount).unwrap()
        );
        let pool_account = StateWithExtensions::<Account>::unpack(&pool_account.data).unwrap();
        assert_eq!(
            pool_account.base.amount,
            u64::try_from(initial_pool - withdraw_amount).unwrap()
        );
        let fee_account =
            StateWithExtensions::<Account>::unpack(&accounts.pool_token_fees_vault_account.data)
                .unwrap();
        assert_eq!(
            fee_account.base.amount,
            TryInto::<u64>::try_into(withdraw_fee).unwrap()
        );
    }

    // todo - elliot - fee account withdrawal
    // // correct withdrawal from fee account
    // {
    //     let (
    //         token_a_key,
    //         mut token_a_account,
    //         token_b_key,
    //         mut token_b_account,
    //         _pool_key,
    //         mut _pool_account,
    //     ) = accounts.setup_token_accounts(&user_key, &withdrawer_key, 0, 0, 0);
    //
    //     let pool_fee_key = accounts.pool_token_fees_vault_key;
    //     let mut pool_fee_account = accounts.pool_token_fees_vault_account.clone();
    //     let fee_account =
    //         StateWithExtensions::<Account>::unpack(&pool_fee_account.data).unwrap();
    //     let pool_fee_amount = fee_account.base.amount;
    //
    //     accounts
    //         .withdraw_all_token_types(
    //             &user_key,
    //             &pool_fee_key,
    //             &mut pool_fee_account,
    //             &token_a_key,
    //             &mut token_a_account,
    //             &token_b_key,
    //             &mut token_b_account,
    //             pool_fee_amount,
    //             0,
    //             0,
    //         )
    //         .unwrap();
    //
    //     let swap_token_a =
    //         StateWithExtensions::<Account>::unpack(&accounts.token_a_vault_account.data)
    //             .unwrap();
    //     let swap_token_b =
    //         StateWithExtensions::<Account>::unpack(&accounts.token_b_vault_account.data)
    //             .unwrap();
    //     let pool_mint =
    //         StateWithExtensions::<Mint>::unpack(&accounts.pool_token_mint_account.data)
    //             .unwrap();
    //     let results = accounts
    //         .swap_curve
    //         .calculator
    //         .pool_tokens_to_trading_tokens(
    //             pool_fee_amount.try_into().unwrap(),
    //             pool_mint.base.supply.try_into().unwrap(),
    //             swap_token_a.base.amount.try_into().unwrap(),
    //             swap_token_b.base.amount.try_into().unwrap(),
    //             RoundDirection::Floor,
    //         )
    //         .unwrap();
    //     let token_a = StateWithExtensions::<Account>::unpack(&token_a_account.data).unwrap();
    //     assert_eq!(
    //         token_a.base.amount,
    //         TryInto::<u64>::try_into(results.token_a_amount).unwrap()
    //     );
    //     let token_b = StateWithExtensions::<Account>::unpack(&token_b_account.data).unwrap();
    //     assert_eq!(
    //         token_b.base.amount,
    //         TryInto::<u64>::try_into(results.token_b_amount).unwrap()
    //     );
    // }
}

#[test_case(spl_token::id(), spl_token::id(), spl_token::id(); "all-token")]
#[test_case(spl_token::id(), spl_token_2022::id(), spl_token_2022::id(); "mixed-pool-token")]
#[test_case(spl_token_2022::id(), spl_token_2022::id(), spl_token_2022::id(); "all-token-2022")]
#[test_case(spl_token_2022::id(), spl_token_2022::id(), spl_token::id(); "a-only-token-2022")]
#[test_case(spl_token_2022::id(), spl_token::id(), spl_token_2022::id(); "b-only-token-2022")]
fn test_withdraw_all_offset_curve(
    pool_token_program_id: Pubkey,
    token_a_program_id: Pubkey,
    token_b_program_id: Pubkey,
) {
    let trade_fee_numerator = 1;
    let trade_fee_denominator = 10;
    let owner_trade_fee_numerator = 1;
    let owner_trade_fee_denominator = 30;
    let owner_withdraw_fee_numerator = 0;
    let owner_withdraw_fee_denominator = 30;
    let host_fee_numerator = 10;
    let host_fee_denominator = 100;

    let token_a_amount = 1_000_000_000;
    let token_b_amount = 10;
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

    let token_b_offset = 2_000_000;
    let curve_params = CurveParameters::Offset { token_b_offset };
    let swap_curve = SwapCurve::new_from_params(curve_params.clone()).unwrap();
    let total_pool = swap_curve.calculator.new_pool_supply();
    let user_key = Pubkey::new_unique();

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

    accounts.initialize_pool().unwrap();

    let (
        token_a_key,
        mut token_a_account,
        token_b_key,
        mut token_b_account,
        _pool_key,
        _pool_account,
    ) = accounts.setup_token_accounts(&user_key, &user_key, 0, 0, 0);

    let pool_key = accounts.admin_authority_pool_token_ata_key;
    let mut pool_account = accounts.admin_authority_pool_token_ata_account.clone();

    // WithdrawAllTokenTypes takes all tokens for A and B.
    // The curve's calculation for token B will say to transfer
    // `token_b_offset + token_b_amount`, but only `token_b_amount` will be
    // moved.
    accounts
        .withdraw_all_token_types(
            &user_key,
            &pool_key,
            &mut pool_account,
            &token_a_key,
            &mut token_a_account,
            &token_b_key,
            &mut token_b_account,
            total_pool.try_into().unwrap(),
            0,
            0,
        )
        .unwrap();

    let token_a = StateWithExtensions::<Account>::unpack(&token_a_account.data).unwrap();
    assert_eq!(token_a.base.amount, token_a_amount);
    let token_b = StateWithExtensions::<Account>::unpack(&token_b_account.data).unwrap();
    assert_eq!(token_b.base.amount, token_b_amount);
    let swap_token_a =
        StateWithExtensions::<Account>::unpack(&accounts.token_a_vault_account.data).unwrap();
    assert_eq!(swap_token_a.base.amount, 0);
    let swap_token_b =
        StateWithExtensions::<Account>::unpack(&accounts.token_b_vault_account.data).unwrap();
    assert_eq!(swap_token_b.base.amount, 0);
}

#[test_case(spl_token::id(), spl_token::id(), spl_token::id(); "all-token")]
#[test_case(spl_token::id(), spl_token_2022::id(), spl_token_2022::id(); "mixed-pool-token")]
#[test_case(spl_token_2022::id(), spl_token_2022::id(), spl_token_2022::id(); "all-token-2022")]
#[test_case(spl_token_2022::id(), spl_token_2022::id(), spl_token::id(); "a-only-token-2022")]
#[test_case(spl_token_2022::id(), spl_token::id(), spl_token_2022::id(); "b-only-token-2022")]
fn test_withdraw_all_constant_price_curve(
    pool_token_program_id: Pubkey,
    token_a_program_id: Pubkey,
    token_b_program_id: Pubkey,
) {
    let trade_fee_numerator = 1;
    let trade_fee_denominator = 10;
    let owner_trade_fee_numerator = 1;
    let owner_trade_fee_denominator = 30;
    let owner_withdraw_fee_numerator = 0;
    let owner_withdraw_fee_denominator = 30;
    let host_fee_numerator = 10;
    let host_fee_denominator = 100;

    // initialize "unbalanced", so that withdrawing all will have some issues
    // A: 1_000_000_000
    // B: 2_000_000_000 (1_000 * 2_000_000)
    let swap_token_a_amount = 1_000_000_000;
    let swap_token_b_amount = 1_000;
    let token_b_price = 2_000_000;
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

    let curve_params = CurveParameters::ConstantPrice { token_b_price };
    let swap_curve = SwapCurve::new_from_params(curve_params.clone()).unwrap();
    let total_pool = swap_curve.calculator.new_pool_supply();
    let user_key = Pubkey::new_unique();
    let withdrawer_key = Pubkey::new_unique();

    let mut accounts = SwapAccountInfo::new(
        &user_key,
        fees,
        SwapTransferFees::default(),
        curve_params,
        InitialSupply {
            initial_supply_a: swap_token_a_amount,
            initial_supply_b: swap_token_b_amount,
        },
        &pool_token_program_id,
        &token_a_program_id,
        &token_b_program_id,
    );

    accounts.initialize_pool().unwrap();

    let (
        token_a_key,
        mut token_a_account,
        token_b_key,
        mut token_b_account,
        _pool_key,
        _pool_account,
    ) = accounts.setup_token_accounts(&user_key, &user_key, 0, 0, 0);

    let pool_key = accounts.admin_authority_pool_token_ata_key;
    let mut pool_account = accounts.admin_authority_pool_token_ata_account.clone();

    // WithdrawAllTokenTypes will not take all token A and B, since their
    // ratio is unbalanced.  It will try to take 1_500_000_000 worth of
    // each token, which means 1_500_000_000 token A, and 750 token B.
    // With no slippage, this will leave 250 token B in the pool.
    assert_eq!(
        Err(SwapError::ExceededSlippage.into()),
        accounts.withdraw_all_token_types(
            &user_key,
            &pool_key,
            &mut pool_account,
            &token_a_key,
            &mut token_a_account,
            &token_b_key,
            &mut token_b_account,
            total_pool.try_into().unwrap(),
            swap_token_a_amount,
            swap_token_b_amount,
        )
    );

    accounts
        .withdraw_all_token_types(
            &user_key,
            &pool_key,
            &mut pool_account,
            &token_a_key,
            &mut token_a_account,
            &token_b_key,
            &mut token_b_account,
            total_pool.try_into().unwrap(),
            0,
            0,
        )
        .unwrap();

    let token_a = StateWithExtensions::<Account>::unpack(&token_a_account.data).unwrap();
    assert_eq!(token_a.base.amount, swap_token_a_amount);
    let token_b = StateWithExtensions::<Account>::unpack(&token_b_account.data).unwrap();
    assert_eq!(token_b.base.amount, 750);
    let swap_token_a =
        StateWithExtensions::<Account>::unpack(&accounts.token_a_vault_account.data).unwrap();
    assert_eq!(swap_token_a.base.amount, 0);
    let swap_token_b =
        StateWithExtensions::<Account>::unpack(&accounts.token_b_vault_account.data).unwrap();
    assert_eq!(swap_token_b.base.amount, 250);

    // deposit now, not enough to cover the tokens already in there
    let token_b_amount = 10;
    let token_a_amount = token_b_amount * token_b_price;
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
        token_a_amount,
        token_b_amount,
        0,
    );

    assert_eq!(
        Err(SwapError::ExceededSlippage.into()),
        accounts.deposit_all_token_types(
            &withdrawer_key,
            &token_a_key,
            &mut token_a_account,
            &token_b_key,
            &mut token_b_account,
            &pool_key,
            &mut pool_account,
            1, // doesn't matter
            token_a_amount,
            token_b_amount,
        )
    );

    // deposit enough tokens, success!
    let token_b_amount = 125;
    let token_a_amount = token_b_amount * token_b_price;
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
        token_a_amount,
        token_b_amount,
        0,
    );

    accounts
        .deposit_all_token_types(
            &withdrawer_key,
            &token_a_key,
            &mut token_a_account,
            &token_b_key,
            &mut token_b_account,
            &pool_key,
            &mut pool_account,
            1, // doesn't matter
            token_a_amount,
            token_b_amount,
        )
        .unwrap();
}
