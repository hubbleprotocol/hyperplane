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
use solana_sdk::account::{
    create_account_for_test, Account as SolanaAccount, ReadableAccount, WritableAccount,
};
use test_case::test_case;

use crate::{
    constraints::SwapConstraints,
    curve::{base::CurveType, fees::Fees},
    error::SwapError,
    instructions::test::runner::{
        processor::{
            do_process_instruction, do_process_instruction_with_fee_constraints, SwapAccountInfo,
            SwapTransferFees,
        },
        token,
    },
    ix,
    model::CurveParameters,
    state::{SwapPool, SwapState},
    utils::seeds,
    InitialSupply,
};

#[test_case(spl_token::id(), spl_token::id(), spl_token::id(); "all-token")]
#[test_case(spl_token::id(), spl_token_2022::id(), spl_token_2022::id(); "mixed-pool-token")]
#[test_case(spl_token_2022::id(), spl_token_2022::id(), spl_token_2022::id(); "all-token-2022")]
#[test_case(spl_token_2022::id(), spl_token_2022::id(), spl_token::id(); "a-only-token-2022")]
#[test_case(spl_token_2022::id(), spl_token::id(), spl_token_2022::id(); "b-only-token-2022")]
fn test_initialize(
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
    let token_b_amount = 2000;
    let pool_token_amount = 10;
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

    // uninitialized token a account
    {
        let old_account = accounts.token_a_vault_account;
        accounts.token_a_vault_account = SolanaAccount::new(0, 0, &token_a_program_id);
        assert_eq!(
            Err(ProgramError::InvalidAccountData),
            accounts.initialize_pool()
        );
        accounts.token_a_vault_account = old_account;
    }

    // uninitialized token b account
    {
        let old_account = accounts.token_b_vault_account;
        accounts.token_b_vault_account = SolanaAccount::new(0, 0, &token_b_program_id);
        assert_eq!(
            Err(ProgramError::InvalidAccountData),
            accounts.initialize_pool()
        );
        accounts.token_b_vault_account = old_account;
    }

    // initialized pool mint
    {
        let old_account = accounts.pool_token_mint_account;
        accounts.pool_token_mint_account = token::create_mint_with_address(
            &accounts.pool_token_mint_key,
            old_account.owner(),
            &Pubkey::new_unique(),
            None,
            None,
            6,
            &TransferFee::default(),
        );
        assert_eq!(
            Err(TokenError::AlreadyInUse.into()),
            accounts.initialize_pool()
        );
        accounts.pool_token_mint_account = old_account;
    }

    // token A account owner is not admin authority
    {
        let fake_admin = Pubkey::new_unique();
        let (_token_a_ata_key, admin_authority_token_a_ata_account) = token::create_token_account(
            &token_a_program_id,
            &accounts.token_a_mint_key,
            &mut accounts.token_a_mint_account,
            &user_key,
            &fake_admin,
            1000,
        );
        let old_account = accounts.admin_authority_token_a_ata_account;
        accounts.admin_authority_token_a_ata_account = admin_authority_token_a_ata_account;
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenOwner.into()
            )),
            accounts.initialize_pool()
        );
        accounts.admin_authority_token_a_ata_account = old_account;
    }

    // token B account owner is not admin authority
    {
        let fake_admin = Pubkey::new_unique();
        let (_token_b_key, token_b_account) = token::create_token_account(
            &token_b_program_id,
            &accounts.token_b_mint_key,
            &mut accounts.token_b_mint_account,
            &user_key,
            &fake_admin,
            0,
        );
        let old_account = accounts.admin_authority_token_b_ata_account;
        accounts.admin_authority_token_b_ata_account = token_b_account;
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::ConstraintTokenOwner.into()
            )),
            accounts.initialize_pool()
        );
        accounts.admin_authority_token_b_ata_account = old_account;
    }

    // pool mint already initialized with freeze authority
    {
        let (_pool_mint_key, pool_mint_account) = token::create_mint(
            &accounts.pool_token_program_id,
            &accounts.pool_authority,
            Some(&user_key),
            None,
            &TransferFee::default(),
        );
        let old_mint = accounts.pool_token_mint_account;
        accounts.pool_token_mint_account = pool_mint_account;
        assert_eq!(
            Err(TokenError::AlreadyInUse.into()),
            accounts.initialize_pool()
        );
        accounts.pool_token_mint_account = old_mint;
    }

    // token A account owned by wrong program
    {
        let (_token_a_key, mut token_a_account) = token::create_token_account(
            &token_a_program_id,
            &accounts.token_a_mint_key,
            &mut accounts.token_a_mint_account,
            &user_key,
            &user_key,
            token_a_amount,
        );
        token_a_account.owner = crate::id();
        let old_account = accounts.admin_authority_token_a_ata_account;
        accounts.admin_authority_token_a_ata_account = token_a_account;
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::AccountOwnedByWrongProgram.into()
            )),
            accounts.initialize_pool()
        );
        accounts.admin_authority_token_a_ata_account = old_account;
    }

    // token B account owned by wrong program
    {
        let (_token_b_key, mut token_b_account) = token::create_token_account(
            &token_b_program_id,
            &accounts.token_b_mint_key,
            &mut accounts.token_b_mint_account,
            &user_key,
            &user_key,
            token_b_amount,
        );
        token_b_account.owner = crate::id();
        let old_account = accounts.admin_authority_token_b_ata_account;
        accounts.admin_authority_token_b_ata_account = token_b_account;
        assert_eq!(
            Err(ProgramError::Custom(
                AnchorError::AccountOwnedByWrongProgram.into()
            )),
            accounts.initialize_pool()
        );
        accounts.admin_authority_token_b_ata_account = old_account;
    }

    // empty token A account
    {
        let (_token_a_key, token_a_account) = token::create_token_account(
            &token_a_program_id,
            &accounts.token_a_mint_key,
            &mut accounts.token_a_mint_account,
            &user_key,
            &user_key,
            0,
        );
        let old_account = accounts.admin_authority_token_a_ata_account;
        accounts.admin_authority_token_a_ata_account = token_a_account;
        assert_eq!(
            Err(TokenError::InsufficientFunds.into()),
            accounts.initialize_pool()
        );
        accounts.admin_authority_token_a_ata_account = old_account;
    }

    // empty token B account
    {
        let (_token_b_key, token_b_account) = token::create_token_account(
            &token_b_program_id,
            &accounts.token_b_mint_key,
            &mut accounts.token_b_mint_account,
            &user_key,
            &user_key,
            0,
        );
        let old_account = accounts.admin_authority_token_b_ata_account;
        accounts.admin_authority_token_b_ata_account = token_b_account;
        assert_eq!(
            Err(TokenError::InsufficientFunds.into()),
            accounts.initialize_pool()
        );
        accounts.admin_authority_token_b_ata_account = old_account;
    }

    // 0 token A initial supply
    {
        let old_initial_supply = accounts.initial_supply;
        accounts.initial_supply = InitialSupply {
            initial_supply_a: 0,
            ..old_initial_supply
        };
        assert_eq!(
            Err(SwapError::EmptySupply.into()),
            accounts.initialize_pool()
        );
        accounts.initial_supply = old_initial_supply;
    }

    // 0 token B initial supply
    {
        let old_initial_supply = accounts.initial_supply;
        accounts.initial_supply = InitialSupply {
            initial_supply_b: 0,
            ..old_initial_supply
        };
        assert_eq!(
            Err(SwapError::EmptySupply.into()),
            accounts.initialize_pool()
        );
        accounts.initial_supply = old_initial_supply;
    }

    // invalid pool tokens
    {
        let old_mint = accounts.pool_token_mint_account;
        let old_pool_account = accounts.admin_authority_pool_token_ata_account;

        let (_pool_mint_key, pool_mint_account) = token::create_mint(
            &accounts.pool_token_program_id,
            &accounts.pool_authority,
            None,
            None,
            &TransferFee::default(),
        );
        accounts.pool_token_mint_account = pool_mint_account;

        let (_empty_pool_token_key, empty_pool_token_account) = token::create_token_account(
            &accounts.pool_token_program_id,
            &accounts.pool_token_mint_key,
            &mut accounts.pool_token_mint_account,
            &accounts.pool_authority,
            &user_key,
            0,
        );

        let (_pool_token_key, pool_token_account) = token::create_token_account(
            &accounts.pool_token_program_id,
            &accounts.pool_token_mint_key,
            &mut accounts.pool_token_mint_account,
            &accounts.pool_authority,
            &user_key,
            pool_token_amount,
        );

        // non-empty pool token account
        accounts.admin_authority_pool_token_ata_account = pool_token_account;
        assert_eq!(
            Err(TokenError::AlreadyInUse.into()),
            accounts.initialize_pool()
        );

        // pool tokens already in circulation
        accounts.admin_authority_pool_token_ata_account = empty_pool_token_account;
        assert_eq!(
            Err(TokenError::AlreadyInUse.into()),
            accounts.initialize_pool()
        );

        accounts.pool_token_mint_account = old_mint;
        accounts.admin_authority_pool_token_ata_account = old_pool_account;
    }

    // pool fee account already initialized
    {
        // wrong mint
        let (pool_mint_key, mut pool_mint_account) = token::create_mint(
            &pool_token_program_id,
            &accounts.pool_authority,
            None,
            None,
            &TransferFee::default(),
        );
        let (_pool_fee_key, pool_fee_account) = token::create_token_account(
            &pool_token_program_id,
            &pool_mint_key,
            &mut pool_mint_account,
            &user_key,
            &user_key,
            0,
        );
        let old_account = accounts.pool_token_fees_vault_account;
        accounts.pool_token_fees_vault_account = pool_fee_account;
        assert_eq!(
            Err(TokenError::AlreadyInUse.into()),
            accounts.initialize_pool()
        );
        accounts.pool_token_fees_vault_account = old_account;
    }

    // wrong pool token program id
    {
        let wrong_pool_token_program_id = Pubkey::new_unique();

        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);
        assert_eq!(
            Err(ProgramError::Custom(AnchorError::InvalidProgramId.into())),
            do_process_instruction(
                ix::initialize_pool(
                    &crate::id(),
                    &accounts.admin_authority,
                    &accounts.pool,
                    &accounts.swap_curve_key,
                    &accounts.token_a_mint_key,
                    &accounts.token_b_mint_key,
                    &accounts.token_a_vault_key,
                    &accounts.token_b_vault_key,
                    &accounts.pool_authority,
                    &accounts.pool_token_mint_key,
                    &accounts.pool_token_fees_vault_key,
                    &accounts.admin_authority_token_a_ata_key,
                    &accounts.admin_authority_token_b_ata_key,
                    &accounts.admin_authority_pool_token_ata_key,
                    &wrong_pool_token_program_id,
                    &accounts.token_a_program_id,
                    &accounts.token_b_program_id,
                    accounts.fees,
                    accounts.initial_supply.clone(),
                    accounts.curve_params.clone().into(),
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
                    &mut accounts.admin_authority_token_a_ata_account,
                    &mut accounts.admin_authority_token_b_ata_account,
                    &mut accounts.admin_authority_pool_token_ata_account,
                    &mut exe.clone(), // system_program
                    &mut create_account_for_test(&Rent::default()),
                    &mut exe.clone(), // pool_token_program
                    &mut exe.clone(), // token_a_program
                    &mut exe.clone(), // token_b_program
                ],
            )
        );
    }

    // create swap with same token A and B
    {
        let (token_b_vault_key, _token_b_vault_bump_seed) = Pubkey::find_program_address(
            &[
                seeds::TOKEN_B_VAULT,
                accounts.pool.as_ref(),
                accounts.token_a_mint_key.as_ref(),
            ],
            &crate::id(),
        );
        let token_b_vault_account = SolanaAccount::new(
            u32::MAX as u64,
            token::get_token_account_space(
                &accounts.token_a_program_id,
                &accounts.token_a_mint_account,
            ), // todo size needed because syscall not stubbed
            &accounts.token_a_program_id, // todo - this should be system but we no-op the system program calls
        );
        let (admin_authority_token_b_ata_key, admin_authority_token_b_ata_account) =
            token::create_token_account(
                &token_a_program_id,
                &accounts.token_a_mint_key,
                &mut accounts.token_a_mint_account,
                &user_key,
                &user_key,
                token_b_amount,
            );
        let old_mint_key = accounts.token_b_mint_key;
        let old_mint = accounts.token_b_mint_account;
        let old_account_key = accounts.token_b_vault_key;
        let old_account = accounts.token_b_vault_account;
        let old_ata_key = accounts.admin_authority_token_b_ata_key;
        let old_token_program = accounts.token_b_program_id;
        let old_ata_account = accounts.admin_authority_token_b_ata_account;
        accounts.token_b_mint_key = accounts.token_a_mint_key;
        accounts.token_b_mint_account = accounts.token_a_mint_account.clone();
        accounts.token_b_vault_key = token_b_vault_key;
        accounts.token_b_vault_account = token_b_vault_account;
        accounts.admin_authority_token_b_ata_key = admin_authority_token_b_ata_key;
        accounts.admin_authority_token_b_ata_account = admin_authority_token_b_ata_account;
        accounts.token_b_program_id = token_a_program_id;
        assert_eq!(
            Err(SwapError::RepeatedMint.into()),
            accounts.initialize_pool()
        );
        accounts.token_b_mint_key = old_mint_key;
        accounts.token_b_mint_account = old_mint;
        accounts.token_b_vault_key = old_account_key;
        accounts.token_b_vault_account = old_account;
        accounts.admin_authority_token_b_ata_key = old_ata_key;
        accounts.admin_authority_token_b_ata_account = old_ata_account;
        accounts.token_b_program_id = old_token_program;
    }

    // create valid swap
    accounts.initialize_pool().unwrap();

    // create invalid flat swap
    {
        let token_b_price = 0;
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
        assert_eq!(
            Err(SwapError::InvalidCurve.into()),
            accounts.initialize_pool()
        );
    }

    // create valid flat swap
    {
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
        let token_b_price = 10_000;
        let curve_params = CurveParameters::ConstantPrice { token_b_price };

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
    }

    // create invalid offset swap
    {
        let token_b_offset = 0;
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
        let curve_params = CurveParameters::Offset { token_b_offset };
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
        assert_eq!(
            Err(SwapError::InvalidCurve.into()),
            accounts.initialize_pool()
        );
    }

    // create valid offset swap
    {
        let token_b_offset = 10;
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

        let curve_params = CurveParameters::Offset { token_b_offset };
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
    }

    // todo - elliot - compile-time constraints
    // // wrong owner key in constraint
    // {
    //     let new_key = Pubkey::new_unique();
    //     let trade_fee_numerator = 25;
    //     let trade_fee_denominator = 10000;
    //     let owner_trade_fee_numerator = 5;
    //     let owner_trade_fee_denominator = 10000;
    //     let host_fee_numerator = 20;
    //     let host_fee_denominator = 100;
    //     let fees = Fees {
    //         trade_fee_numerator,
    //         trade_fee_denominator,
    //         owner_trade_fee_numerator,
    //         owner_trade_fee_denominator,
    //         owner_withdraw_fee_numerator,
    //         owner_withdraw_fee_denominator,
    //         host_fee_numerator,
    //         host_fee_denominator,
    //     };
    //     let curve_params = CurveParameters::ConstantProduct;
    //     let owner_key = &new_key.to_string();
    //     let valid_curve_types = &[CurveType::ConstantProduct];
    //     let constraints = Some(SwapConstraints {
    //         owner_key,
    //         valid_curve_types,
    //         fees: &fees,
    //     });
    //     let mut accounts = SwapAccountInfo::new(
    //         &user_key,
    //         fees.clone(),
    //         SwapTransferFees::default(),
    //         curve_params.clone(),
    //         token_a_amount,
    //         token_b_amount,
    //         &token_a_program_id,
    //         &token_b_program_id,
    //     );
    //     let exe = &mut SolanaAccount::default();
    //     exe.set_executable(true);
    //     assert_eq!(
    //         Err(SwapError::InvalidOwner.into()),
    //         do_process_instruction_with_fee_constraints(
    //             ix::initialize_pool(
    //                 &crate::id(),
    //                 &accounts.admin_authority,
    //                 &accounts.pool,
    //                 &accounts.swap_curve_key,
    //                 &accounts.token_a_mint_key,
    //                 &accounts.token_b_mint_key,
    //                 &accounts.token_a_vault_key,
    //                 &accounts.token_b_vault_key,
    //                 &accounts.pool_authority,
    //                 &accounts.pool_token_mint_key,
    //                 &accounts.pool_token_fees_vault_key,
    //                 &accounts.admin_authority_pool_token_ata_key,
    //                 accounts.fees.clone(),
    //                 accounts.curve_params.clone(),
    //             )
    //             .unwrap(),
    //             vec![
    //                 &mut SolanaAccount::default(),
    //                 &mut accounts.pool_account,
    //                 &mut accounts.swap_curve_account,
    //                 &mut SolanaAccount::default(),
    //                 &mut accounts.token_a_mint_account,
    //                 &mut accounts.token_b_mint_account,
    //                 &mut accounts.token_a_vault_account,
    //                 &mut accounts.token_b_vault_account,
    //                 &mut accounts.pool_token_mint_account,
    //                 &mut accounts.pool_token_fees_vault_account,
    //                 &mut accounts.admin_authority_pool_token_ata_account,
    //                 &mut exe.clone(),
    //                 &mut create_account_for_test(&Rent::default()),
    //                 &mut exe.clone(),
    //             ],
    //             &constraints,
    //         )
    //     );
    // }
    //
    // // wrong fee in constraint
    // {
    //     let trade_fee_numerator = 25;
    //     let trade_fee_denominator = 10000;
    //     let owner_trade_fee_numerator = 5;
    //     let owner_trade_fee_denominator = 10000;
    //     let host_fee_numerator = 20;
    //     let host_fee_denominator = 100;
    //     let fees = Fees {
    //         trade_fee_numerator,
    //         trade_fee_denominator,
    //         owner_trade_fee_numerator,
    //         owner_trade_fee_denominator,
    //         owner_withdraw_fee_numerator,
    //         owner_withdraw_fee_denominator,
    //         host_fee_numerator,
    //         host_fee_denominator,
    //     };
    //
    //     let curve_params = CurveParameters::ConstantProduct;
    //     let owner_key = &user_key.to_string();
    //     let valid_curve_types = &[CurveType::ConstantProduct];
    //     let constraints = Some(SwapConstraints {
    //         owner_key,
    //         valid_curve_types,
    //         fees: &fees,
    //     });
    //     let mut bad_fees = fees.clone();
    //     bad_fees.trade_fee_numerator = trade_fee_numerator - 1;
    //     let mut accounts = SwapAccountInfo::new(
    //         &user_key,
    //         bad_fees,
    //         SwapTransferFees::default(),
    //         curve_params,
    //         token_a_amount,
    //         token_b_amount,
    //         &token_a_program_id,
    //         &token_b_program_id,
    //     );
    //     let exe = &mut SolanaAccount::default();
    //     exe.set_executable(true);
    //     assert_eq!(
    //         Err(SwapError::InvalidFee.into()),
    //         do_process_instruction_with_fee_constraints(
    //             ix::initialize_pool(
    //                 &crate::id(),
    //                 &accounts.admin_authority,
    //                 &accounts.pool,
    //                 &accounts.swap_curve_key,
    //                 &accounts.token_a_mint_key,
    //                 &accounts.token_b_mint_key,
    //                 &accounts.token_a_vault_key,
    //                 &accounts.token_b_vault_key,
    //                 &accounts.pool_authority,
    //                 &accounts.pool_token_mint_key,
    //                 &accounts.pool_token_fees_vault_key,
    //                 &accounts.admin_authority_pool_token_ata_key,
    //                 accounts.fees.clone(),
    //                 accounts.curve_params.clone(),
    //             )
    //             .unwrap(),
    //             vec![
    //                 &mut SolanaAccount::default(),
    //                 &mut accounts.pool_account,
    //                 &mut accounts.swap_curve_account,
    //                 &mut SolanaAccount::default(),
    //                 &mut accounts.token_a_mint_account,
    //                 &mut accounts.token_b_mint_account,
    //                 &mut accounts.token_a_vault_account,
    //                 &mut accounts.token_b_vault_account,
    //                 &mut accounts.pool_token_mint_account,
    //                 &mut accounts.pool_token_fees_vault_account,
    //                 &mut accounts.admin_authority_pool_token_ata_account,
    //                 &mut exe.clone(),
    //                 &mut create_account_for_test(&Rent::default()),
    //                 &mut exe.clone(),
    //             ],
    //             &constraints,
    //         )
    //     );
    // }

    // create valid swap with constraints
    {
        let trade_fee_numerator = 25;
        let trade_fee_denominator = 10000;
        let owner_trade_fee_numerator = 5;
        let owner_trade_fee_denominator = 10000;
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
        let curve_params = CurveParameters::ConstantProduct;
        let owner_key = &user_key.to_string();
        let valid_curve_types = &[CurveType::ConstantProduct];
        let constraints = Some(SwapConstraints {
            owner_key,
            valid_curve_types,
            fees: &fees,
        });
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
        let exe = &mut SolanaAccount::default();
        exe.set_executable(true);
        do_process_instruction_with_fee_constraints(
            crate::ix::initialize_pool(
                &crate::id(),
                &accounts.admin_authority,
                &accounts.pool,
                &accounts.swap_curve_key,
                &accounts.token_a_mint_key,
                &accounts.token_b_mint_key,
                &accounts.token_a_vault_key,
                &accounts.token_b_vault_key,
                &accounts.pool_authority,
                &accounts.pool_token_mint_key,
                &accounts.pool_token_fees_vault_key,
                &accounts.admin_authority_token_a_ata_key,
                &accounts.admin_authority_token_b_ata_key,
                &accounts.admin_authority_pool_token_ata_key,
                &accounts.pool_token_program_id,
                &accounts.token_a_program_id,
                &accounts.token_b_program_id,
                accounts.fees,
                accounts.initial_supply,
                accounts.curve_params.into(),
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
                &mut accounts.admin_authority_token_a_ata_account,
                &mut accounts.admin_authority_token_b_ata_account,
                &mut accounts.admin_authority_pool_token_ata_account,
                &mut exe.clone(), // system_program
                &mut create_account_for_test(&Rent::default()),
                &mut exe.clone(), // pool_token_program
                &mut exe.clone(), // token_a_program
                &mut exe.clone(), // token_b_program
            ],
            &constraints,
        )
        .unwrap();
    }

    // create again
    {
        assert_eq!(
            Err(TokenError::AlreadyInUse.into()),
            accounts.initialize_pool()
        );
    }

    let mut data = accounts.pool_account.data.as_ref();
    let swap_pool: SwapPool = AccountDeserialize::try_deserialize(&mut data).unwrap();
    assert!(swap_pool.is_initialized());
    assert_eq!(swap_pool.bump_seed(), accounts.pool_authority_bump_seed);
    assert_eq!(swap_pool.pool_authority, accounts.pool_authority);
    assert_eq!(swap_pool.curve_type(), accounts.swap_curve.curve_type);
    assert_eq!(swap_pool.swap_curve, accounts.swap_curve_key);
    assert_eq!(*swap_pool.token_a_account(), accounts.token_a_vault_key);
    assert_eq!(*swap_pool.token_b_account(), accounts.token_b_vault_key);
    assert_eq!(*swap_pool.pool_mint(), accounts.pool_token_mint_key);
    assert_eq!(*swap_pool.token_a_mint(), accounts.token_a_mint_key);
    assert_eq!(*swap_pool.token_b_mint(), accounts.token_b_mint_key);
    assert_eq!(
        *swap_pool.pool_fee_account(),
        accounts.pool_token_fees_vault_key
    );
    assert_eq!(swap_pool.fees, accounts.fees);
    let token_a =
        StateWithExtensions::<Account>::unpack(&accounts.token_a_vault_account.data).unwrap();
    assert_eq!(token_a.base.amount, token_a_amount);
    let token_b =
        StateWithExtensions::<Account>::unpack(&accounts.token_b_vault_account.data).unwrap();
    assert_eq!(token_b.base.amount, token_b_amount);
    let pool_account = StateWithExtensions::<Account>::unpack(
        &accounts.admin_authority_pool_token_ata_account.data,
    )
    .unwrap();
    let pool_mint =
        StateWithExtensions::<Mint>::unpack(&accounts.pool_token_mint_account.data).unwrap();
    assert_eq!(pool_mint.base.supply, pool_account.base.amount);
}
