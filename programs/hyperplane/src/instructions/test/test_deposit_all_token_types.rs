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
    curve::{calculator::INITIAL_SWAP_POOL_AMOUNT, fees::Fees},
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
fn test_deposit(
    pool_token_program_id: Pubkey,
    token_a_program_id: Pubkey,
    token_b_program_id: Pubkey,
) {
    let user_key = Pubkey::new_unique();
    let depositor_key = Pubkey::new_unique();
    let trade_fee_numerator = 1;
    let trade_fee_denominator = 2;
    let owner_trade_fee_numerator = 1;
    let owner_trade_fee_denominator = 10;
    let owner_withdraw_fee_numerator = 1;
    let owner_withdraw_fee_denominator = 5;
    let host_fee_numerator = 20;
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
    let token_b_amount = 9000;
    let curve_params = CurveParameters::ConstantProduct;

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

    // depositing 10% of the current pool amount in token A and B means
    // that our pool tokens will be worth 1 / 10 of the current pool amount
    let pool_amount = INITIAL_SWAP_POOL_AMOUNT / 10;
    let deposit_a = token_a_amount / 10;
    let deposit_b = token_b_amount / 10;

    // swap not initialized
    {
        let (token_a_key, mut token_a_account) = token::create_token_account(
            &accounts.token_a_program_id,
            &accounts.token_a_mint_key,
            &mut accounts.token_a_mint_account,
            &user_key,
            &depositor_key,
            deposit_a,
        );
        let (token_b_key, mut token_b_account) = token::create_token_account(
            &accounts.token_b_program_id,
            &accounts.token_b_mint_key,
            &mut accounts.token_b_mint_account,
            &user_key,
            &depositor_key,
            deposit_b,
        );
        // use token A mint because pool mint not initialized
        let (pool_key, mut pool_account) = token::create_token_account(
            &accounts.token_a_program_id,
            &accounts.token_a_mint_key,
            &mut accounts.token_a_mint_account,
            &user_key,
            &depositor_key,
            0,
        );
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::AccountDiscriminatorMismatch.into()
            )),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );
    }

    accounts.initialize_pool().unwrap();

    // wrong owner for pool account
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        let old_pool_account = accounts.pool_account;
        let mut wrong_pool_account = old_pool_account.clone();
        wrong_pool_account.owner = Pubkey::new_unique();
        accounts.pool_account = wrong_pool_account;
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::AccountOwnedByWrongProgram.into()
            )),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );
        accounts.pool_account = old_pool_account;
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
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        let old_authority = accounts.pool_authority;
        let (bad_authority_key, _bump_seed) = Pubkey::find_program_address(
            &[seeds::POOL_AUTHORITY, accounts.pool.as_ref()],
            &accounts.pool_token_program_id,
        );
        accounts.pool_authority = bad_authority_key;
        assert_eq!(
            Err(SwapError::InvalidProgramAddress.into()),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );
        accounts.pool_authority = old_authority;
    }

    // not enough token A
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a / 2, deposit_b, 0);
        assert_eq!(
            Err(TokenError::InsufficientFunds.into()),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );
    }

    // not enough token B
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b / 2, 0);
        assert_eq!(
            Err(TokenError::InsufficientFunds.into()),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
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
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);

        let old_token_a_program = accounts.token_a_program_id;
        let old_token_a_mint_account = accounts.token_a_mint_account;
        let old_token_a_mint_key = accounts.token_a_mint_key;
        accounts.token_a_program_id = accounts.token_b_program_id;
        accounts.token_a_mint_key = accounts.token_b_mint_key;
        accounts.token_a_mint_account = accounts.token_b_mint_account;
        accounts.token_b_program_id = old_token_a_program;
        accounts.token_b_mint_key = old_token_a_mint_key;
        accounts.token_b_mint_account = old_token_a_mint_account;
        assert_eq!(
            Err(ProgramError::Custom(AnchorError::ConstraintHasOne.into())),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_b_key,
                &mut token_b_account,
                &token_a_key,
                &mut token_a_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );
        let old_token_b_program = accounts.token_a_program_id;
        let old_token_b_mint_account = accounts.token_a_mint_account;
        let old_token_b_mint_key = accounts.token_a_mint_key;
        accounts.token_a_program_id = accounts.token_b_program_id;
        accounts.token_a_mint_key = accounts.token_b_mint_key;
        accounts.token_a_mint_account = accounts.token_b_mint_account;
        accounts.token_b_program_id = old_token_b_program;
        accounts.token_b_mint_key = old_token_b_mint_key;
        accounts.token_b_mint_account = old_token_b_mint_account;
    }

    // wrong pool token account
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            _pool_key,
            mut _pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        let (
            wrong_token_key,
            mut wrong_token_account,
            _token_b_key,
            mut _token_b_account,
            _pool_key,
            _pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenMint.into()
            )),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &wrong_token_key,
                &mut wrong_token_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );
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
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        let user_transfer_authority_key = Pubkey::new_unique();
        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);
        assert_eq!(
            Err(TokenError::OwnerMismatch.into()),
            do_process_instruction(
                ix::deposit_all_token_types(
                    &crate::id(),
                    &token_a_program_id,
                    &token_b_program_id,
                    &accounts.pool_token_program_id,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &user_transfer_authority_key,
                    &token_a_key,
                    &token_b_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &accounts.pool_token_mint_key,
                    &pool_key,
                    &accounts.token_a_mint_key,
                    &accounts.token_b_mint_key,
                    &accounts.swap_curve_key,
                    ix::DepositAllTokenTypes {
                        pool_token_amount: pool_amount.try_into().unwrap(),
                        maximum_token_a_amount: deposit_a,
                        maximum_token_b_amount: deposit_b,
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

    // wrong token a program id
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        let wrong_key = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);

        assert_eq!(
            Err(ProgramError::Custom(AnchorError::InvalidProgramId.into())),
            do_process_instruction(
                ix::deposit_all_token_types(
                    &crate::id(),
                    &wrong_key,
                    &accounts.token_b_program_id,
                    &accounts.pool_token_program_id,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &accounts.pool_authority,
                    &token_a_key,
                    &token_b_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &accounts.pool_token_mint_key,
                    &pool_key,
                    &accounts.token_a_mint_key,
                    &accounts.token_b_mint_key,
                    &accounts.swap_curve_key,
                    ix::DepositAllTokenTypes {
                        pool_token_amount: pool_amount.try_into().unwrap(),
                        maximum_token_a_amount: deposit_a,
                        maximum_token_b_amount: deposit_b,
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
                    &mut token_a_account,
                    &mut token_b_account,
                    &mut pool_account,
                    &mut exe.clone(),
                    &mut exe.clone(),
                    &mut exe.clone(),
                ],
            )
        );
    }

    // wrong token b program id
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        let wrong_key = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);

        assert_eq!(
            Err(ProgramError::Custom(AnchorError::InvalidProgramId.into())),
            do_process_instruction(
                ix::deposit_all_token_types(
                    &crate::id(),
                    &accounts.token_a_program_id,
                    &wrong_key,
                    &accounts.pool_token_program_id,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &accounts.pool_authority,
                    &token_a_key,
                    &token_b_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &accounts.pool_token_mint_key,
                    &pool_key,
                    &accounts.token_a_mint_key,
                    &accounts.token_b_mint_key,
                    &accounts.swap_curve_key,
                    ix::DepositAllTokenTypes {
                        pool_token_amount: pool_amount.try_into().unwrap(),
                        maximum_token_a_amount: deposit_a,
                        maximum_token_b_amount: deposit_b,
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
                    &mut token_a_account,
                    &mut token_b_account,
                    &mut pool_account,
                    &mut exe.clone(),
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
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        let wrong_key = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);

        assert_eq!(
            Err(ProgramError::Custom(AnchorError::InvalidProgramId.into())),
            do_process_instruction(
                ix::deposit_all_token_types(
                    &crate::id(),
                    &accounts.token_a_program_id,
                    &accounts.token_b_program_id,
                    &wrong_key,
                    &accounts.pool,
                    &accounts.pool_authority,
                    &accounts.pool_authority,
                    &token_a_key,
                    &token_b_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &accounts.pool_token_mint_key,
                    &pool_key,
                    &accounts.token_a_mint_key,
                    &accounts.token_b_mint_key,
                    &accounts.swap_curve_key,
                    ix::DepositAllTokenTypes {
                        pool_token_amount: pool_amount.try_into().unwrap(),
                        maximum_token_a_amount: deposit_a,
                        maximum_token_b_amount: deposit_b,
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
                    &mut token_a_account,
                    &mut token_b_account,
                    &mut pool_account,
                    &mut exe.clone(),
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
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);

        let old_a_key = accounts.token_a_vault_key;
        let old_a_account = accounts.token_a_vault_account;

        accounts.token_a_vault_key = token_a_key;
        accounts.token_a_vault_account = token_a_account.clone();

        // wrong swap token a vault account
        assert_eq!(
            Err(SwapError::IncorrectSwapAccount.into()),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );

        accounts.token_a_vault_key = old_a_key;
        accounts.token_a_vault_account = old_a_account;

        let old_b_key = accounts.token_b_vault_key;
        let old_b_account = accounts.token_b_vault_account;

        accounts.token_b_vault_key = token_b_key;
        accounts.token_b_vault_account = token_b_account.clone();

        // wrong swap token b vault account
        assert_eq!(
            Err(SwapError::IncorrectSwapAccount.into()),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
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
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
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
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );

        accounts.pool_token_mint_key = old_pool_key;
        accounts.pool_token_mint_account = old_pool_account;
    }

    // deposit 1 pool token fails beacuse it equates to 0 swap tokens
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        assert_eq!(
            Err(SwapError::ZeroTradingTokens.into()),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                1,
                deposit_a,
                deposit_b,
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
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        // maximum A amount in too low
        assert_eq!(
            Err(SwapError::ExceededSlippage.into()),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a / 10,
                deposit_b,
            )
        );
        // maximum B amount in too low
        assert_eq!(
            Err(SwapError::ExceededSlippage.into()),
            accounts.deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b / 10,
            )
        );
    }

    // invalid input: can't use swap pool tokens as source
    {
        let (
            _token_a_key,
            _token_a_account,
            _token_b_key,
            _token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        let swap_token_a_key = accounts.token_a_vault_key;
        let mut swap_token_a_account = accounts.get_token_account(&swap_token_a_key).clone();
        let swap_token_b_key = accounts.token_b_vault_key;
        let mut swap_token_b_account = accounts.get_token_account(&swap_token_b_key).clone();
        let authority_key = accounts.pool_authority;
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenOwner.into()
            )),
            accounts.deposit_all_token_types(
                &authority_key,
                &swap_token_a_key,
                &mut swap_token_a_account,
                &swap_token_b_key,
                &mut swap_token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
        );
    }

    // correctly deposit
    {
        let (
            token_a_key,
            mut token_a_account,
            token_b_key,
            mut token_b_account,
            pool_key,
            mut pool_account,
        ) = accounts.setup_token_accounts(&user_key, &depositor_key, deposit_a, deposit_b, 0);
        accounts
            .deposit_all_token_types(
                &depositor_key,
                &token_a_key,
                &mut token_a_account,
                &token_b_key,
                &mut token_b_account,
                &pool_key,
                &mut pool_account,
                pool_amount.try_into().unwrap(),
                deposit_a,
                deposit_b,
            )
            .unwrap();

        let swap_token_a =
            StateWithExtensions::<Account>::unpack(&accounts.token_a_vault_account.data).unwrap();
        assert_eq!(swap_token_a.base.amount, deposit_a + token_a_amount);
        let swap_token_b =
            StateWithExtensions::<Account>::unpack(&accounts.token_b_vault_account.data).unwrap();
        assert_eq!(swap_token_b.base.amount, deposit_b + token_b_amount);
        let token_a = StateWithExtensions::<Account>::unpack(&token_a_account.data).unwrap();
        assert_eq!(token_a.base.amount, 0);
        let token_b = StateWithExtensions::<Account>::unpack(&token_b_account.data).unwrap();
        assert_eq!(token_b.base.amount, 0);
        let pool_account = StateWithExtensions::<Account>::unpack(&pool_account.data).unwrap();
        let swap_pool_account = StateWithExtensions::<Account>::unpack(
            &accounts.admin_authority_pool_token_ata_account.data,
        )
        .unwrap();
        let pool_mint =
            StateWithExtensions::<Mint>::unpack(&accounts.pool_token_mint_account.data).unwrap();
        assert_eq!(
            pool_mint.base.supply,
            pool_account.base.amount + swap_pool_account.base.amount
        );
    }
}
