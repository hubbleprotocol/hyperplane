//! Helpers for working with swaps in a fuzzing environment

use hyperplane::{
    curve::{base::SwapCurve, calculator::TradeDirection, fees::Fees},
    instructions::model::CurveParameters,
    ix::{
        self, DepositAllTokenTypes, DepositSingleTokenTypeExactAmountIn, Swap,
        WithdrawAllTokenTypes, WithdrawSingleTokenTypeExactAmountOut,
    },
    state::{Curve, SwapPool},
    utils::seeds,
    InitialSupply,
};
use solana_program::{
    bpf_loader, entrypoint::ProgramResult, program_pack::Pack, pubkey::Pubkey, rent::Rent,
    system_program, sysvar::Sysvar,
};
use solana_sdk::account::create_account_for_test;
use spl_token_2022::instruction::approve;

use crate::{
    native_account_data::NativeAccountData, native_processor::do_process_instruction, native_token,
    native_token::get_token_account_space,
};

pub struct NativeTokenSwap {
    pub admin_authority: NativeAccountData,
    pub pool_authority_bump_seed: u8,
    pub pool_authority_account: NativeAccountData,
    pub fees: Fees,
    pub swap_curve: SwapCurve,
    pub pool_account: NativeAccountData,
    pub swap_curve_account: NativeAccountData,
    pub pool_token_mint_account: NativeAccountData,
    pub pool_token_fees_vault_account: NativeAccountData,
    pub admin_authority_token_a_ata: NativeAccountData,
    pub admin_authority_token_b_ata: NativeAccountData,
    pub admin_authority_pool_token_ata: NativeAccountData,
    pub token_a_account: NativeAccountData,
    pub token_a_mint_account: NativeAccountData,
    pub token_b_account: NativeAccountData,
    pub token_b_mint_account: NativeAccountData,
    pub pool_token_program_account: NativeAccountData,
    pub token_a_program_account: NativeAccountData,
    pub token_b_program_account: NativeAccountData,
}

pub fn create_program_account(program_id: Pubkey) -> NativeAccountData {
    let mut account_data = NativeAccountData::new(0, bpf_loader::id());
    account_data.key = program_id;
    account_data.executable = true;
    account_data
}

pub fn create_sysvar_account<S: Sysvar>(sysvar: &S) -> NativeAccountData {
    let account = create_account_for_test(sysvar);

    NativeAccountData {
        key: S::id(),
        lamports: account.lamports,
        data: account.data,
        program_id: account.owner,
        is_signer: false,
        executable: account.executable,
    }
}

impl NativeTokenSwap {
    pub fn new(
        fees: Fees,
        curve_params: CurveParameters,
        token_a_amount: u64,
        token_b_amount: u64,
    ) -> Self {
        let (token_a_decimals, token_b_decimals) = match curve_params {
            CurveParameters::Stable {
                token_a_decimals,
                token_b_decimals,
                ..
            } => (token_a_decimals, token_b_decimals),
            _ => (6, 6),
        };
        let mut admin_authority = NativeAccountData::new(0, system_program::id());
        admin_authority.is_signer = true;
        let mut pool_account = NativeAccountData::new(SwapPool::LEN, hyperplane::id());
        let (swap_curve_key, _swap_curve_bump_seed) = Pubkey::find_program_address(
            &[seeds::SWAP_CURVE, pool_account.key.as_ref()],
            &hyperplane::id(),
        );
        let mut swap_curve_account =
            NativeAccountData::new_with_key(swap_curve_key, Curve::LEN, hyperplane::id());
        let (pool_authority_key, pool_authority_bump_seed) = Pubkey::find_program_address(
            &[seeds::POOL_AUTHORITY, pool_account.key.as_ref()],
            &hyperplane::id(),
        );
        let mut pool_authority_account = create_program_account(pool_authority_key);
        let mut system_program_account = create_program_account(system_program::id());
        let mut rent = create_sysvar_account(&Rent::default());
        let mut pool_token_program_account = create_program_account(spl_token_2022::id());
        let mut token_b_program_account = create_program_account(spl_token::id());
        let mut token_a_program_account = create_program_account(spl_token::id());

        let (pool_token_mint_key, _pool_token_mint_bump_seed) = Pubkey::find_program_address(
            &[seeds::POOL_TOKEN_MINT, pool_account.key.as_ref()],
            &hyperplane::id(),
        );
        let mut pool_token_mint_account = NativeAccountData::new_with_key(
            pool_token_mint_key,
            spl_token_2022::state::Mint::LEN,
            spl_token_2022::id(), // todo - this should be system but we no-op the system program calls
        );

        let mut admin_authority_pool_token_ata =
            NativeAccountData::new(spl_token_2022::state::Account::LEN, spl_token_2022::id());

        let (pool_token_fees_vault_key, _pool_token_fees_vault_bump_seed) =
            Pubkey::find_program_address(
                &[
                    seeds::POOL_TOKEN_FEES_VAULT,
                    pool_account.key.as_ref(),
                    pool_token_mint_key.as_ref(),
                ],
                &hyperplane::id(),
            );
        let mut pool_token_fees_vault_account = NativeAccountData::new_with_key(
            pool_token_fees_vault_key,
            spl_token_2022::state::Account::LEN,
            spl_token_2022::id(),
        );

        let mut token_a_mint_account =
            native_token::create_mint(&admin_authority.key, token_a_decimals);
        let (token_a_vault_key, _token_a_vault_bump_seed) = Pubkey::find_program_address(
            &[
                seeds::TOKEN_A_VAULT,
                pool_account.key.as_ref(),
                token_a_mint_account.key.as_ref(),
            ],
            &hyperplane::id(),
        );
        let mut token_a_vault_account = NativeAccountData::new_with_key(
            token_a_vault_key,
            get_token_account_space(&token_a_program_account.key, &token_a_mint_account),
            token_a_program_account.key,
        );
        let mut admin_authority_token_a_ata_account = native_token::create_token_account(
            &mut token_a_mint_account,
            &token_a_program_account.key,
            &admin_authority.key,
            token_a_amount,
        );
        let mut token_b_mint_account =
            native_token::create_mint(&admin_authority.key, token_b_decimals);
        let (token_b_vault_key, _token_b_vault_bump_seed) = Pubkey::find_program_address(
            &[
                seeds::TOKEN_B_VAULT,
                pool_account.key.as_ref(),
                token_b_mint_account.key.as_ref(),
            ],
            &hyperplane::id(),
        );
        let mut token_b_vault_account = NativeAccountData::new_with_key(
            token_b_vault_key,
            get_token_account_space(&token_b_program_account.key, &token_b_mint_account),
            token_b_program_account.key,
        );
        let mut admin_authority_token_b_ata_account = native_token::create_token_account(
            &mut token_b_mint_account,
            &token_b_program_account.key,
            &admin_authority.key,
            token_b_amount,
        );

        let init_instruction = ix::initialize_pool(
            &hyperplane::id(),
            &admin_authority.key,
            &pool_account.key,
            &swap_curve_account.key,
            &token_a_mint_account.key,
            &token_b_mint_account.key,
            &token_a_vault_account.key,
            &token_b_vault_account.key,
            &pool_authority_account.key,
            &pool_token_mint_account.key,
            &pool_token_fees_vault_account.key,
            &admin_authority_token_a_ata_account.key,
            &admin_authority_token_b_ata_account.key,
            &admin_authority_pool_token_ata.key,
            &spl_token_2022::id(),
            &token_a_program_account.key,
            &token_b_program_account.key,
            fees,
            InitialSupply {
                initial_supply_a: token_a_amount,
                initial_supply_b: token_b_amount,
            },
            curve_params.clone().into(),
        )
        .unwrap();

        do_process_instruction(
            init_instruction,
            &[
                admin_authority.as_account_info(),
                pool_account.as_account_info(),
                swap_curve_account.as_account_info(),
                pool_authority_account.as_account_info(),
                token_a_mint_account.as_account_info(),
                token_b_mint_account.as_account_info(),
                token_a_vault_account.as_account_info(),
                token_b_vault_account.as_account_info(),
                pool_token_mint_account.as_account_info(),
                pool_token_fees_vault_account.as_account_info(),
                admin_authority_token_a_ata_account.as_account_info(),
                admin_authority_token_b_ata_account.as_account_info(),
                admin_authority_pool_token_ata.as_account_info(),
                system_program_account.as_account_info(),
                rent.as_account_info(),
                pool_token_program_account.as_account_info(),
                token_a_program_account.as_account_info(),
                token_b_program_account.as_account_info(),
            ],
        )
        .unwrap();

        Self {
            admin_authority,
            pool_authority_bump_seed,
            pool_authority_account,
            fees,
            pool_account,
            swap_curve: SwapCurve::new_from_params(curve_params).unwrap(),
            swap_curve_account,
            pool_token_mint_account,
            pool_token_fees_vault_account,
            admin_authority_token_a_ata: admin_authority_token_a_ata_account,
            admin_authority_token_b_ata: admin_authority_token_b_ata_account,
            admin_authority_pool_token_ata,
            token_a_account: token_a_vault_account,
            token_a_mint_account,
            token_b_account: token_b_vault_account,
            token_b_mint_account,
            pool_token_program_account,
            token_a_program_account,
            token_b_program_account,
        }
    }

    pub fn create_pool_account(&mut self) -> NativeAccountData {
        native_token::create_token_account(
            &mut self.pool_token_mint_account,
            &self.pool_token_program_account.key,
            &self.admin_authority.key,
            0,
        )
    }

    pub fn create_token_a_account(&mut self, amount: u64) -> NativeAccountData {
        native_token::create_token_account(
            &mut self.token_a_mint_account,
            &self.token_a_program_account.key,
            &self.admin_authority.key,
            amount,
        )
    }

    pub fn create_token_b_account(&mut self, amount: u64) -> NativeAccountData {
        native_token::create_token_account(
            &mut self.token_b_mint_account,
            &self.token_b_program_account.key,
            &self.admin_authority.key,
            amount,
        )
    }

    pub fn swap_a_to_b(
        &mut self,
        token_a_account: &mut NativeAccountData,
        token_b_account: &mut NativeAccountData,
        instruction: Swap,
    ) -> ProgramResult {
        let mut user_transfer_account = NativeAccountData::new(0, system_program::id());
        user_transfer_account.is_signer = true;
        do_process_instruction(
            approve(
                &self.token_a_program_account.key,
                &token_a_account.key,
                &user_transfer_account.key,
                &self.admin_authority.key,
                &[],
                instruction.amount_in,
            )
            .unwrap(),
            &[
                token_a_account.as_account_info(),
                user_transfer_account.as_account_info(),
                self.admin_authority.as_account_info(),
            ],
        )
        .unwrap();
        let swap_instruction = ix::swap(
            &hyperplane::id(),
            &spl_token::id(),
            &spl_token::id(),
            &spl_token_2022::id(),
            &self.pool_account.key,
            &self.pool_authority_account.key,
            &user_transfer_account.key,
            &token_a_account.key,
            &self.token_a_account.key,
            &self.token_b_account.key,
            &token_b_account.key,
            &self.pool_token_mint_account.key,
            &self.pool_token_fees_vault_account.key,
            &self.token_a_mint_account.key,
            &self.token_b_mint_account.key,
            &self.swap_curve_account.key,
            Some(&self.admin_authority_pool_token_ata.key),
            instruction,
        )
        .unwrap();

        do_process_instruction(
            swap_instruction,
            &[
                self.admin_authority.as_account_info(),
                self.pool_account.as_account_info(),
                self.swap_curve_account.as_account_info(),
                self.pool_authority_account.as_account_info(),
                self.token_a_mint_account.as_account_info(),
                self.token_b_mint_account.as_account_info(),
                self.token_a_account.as_account_info(),
                self.token_b_account.as_account_info(),
                self.pool_token_mint_account.as_account_info(),
                self.pool_token_fees_vault_account.as_account_info(),
                token_a_account.as_account_info(),
                token_b_account.as_account_info(),
                self.admin_authority_pool_token_ata.as_account_info(),
                self.pool_token_program_account.as_account_info(),
                self.token_a_program_account.as_account_info(),
                self.token_b_program_account.as_account_info(),
            ],
        )
    }

    pub fn swap_b_to_a(
        &mut self,
        token_b_account: &mut NativeAccountData,
        token_a_account: &mut NativeAccountData,
        instruction: Swap,
    ) -> ProgramResult {
        let mut user_transfer_account = NativeAccountData::new(0, system_program::id());
        user_transfer_account.is_signer = true;
        do_process_instruction(
            approve(
                &self.token_b_program_account.key,
                &token_b_account.key,
                &user_transfer_account.key,
                &self.admin_authority.key,
                &[],
                instruction.amount_in,
            )
            .unwrap(),
            &[
                token_b_account.as_account_info(),
                user_transfer_account.as_account_info(),
                self.admin_authority.as_account_info(),
            ],
        )
        .unwrap();

        let swap_instruction = ix::swap(
            &hyperplane::id(),
            &spl_token::id(),
            &spl_token::id(),
            &spl_token_2022::id(),
            &self.pool_account.key,
            &self.pool_authority_account.key,
            &user_transfer_account.key,
            &token_b_account.key,
            &self.token_b_account.key,
            &self.token_a_account.key,
            &token_a_account.key,
            &self.pool_token_mint_account.key,
            &self.pool_token_fees_vault_account.key,
            &self.token_b_mint_account.key,
            &self.token_a_mint_account.key,
            &self.swap_curve_account.key,
            Some(&self.admin_authority_pool_token_ata.key),
            instruction,
        )
        .unwrap();

        do_process_instruction(
            swap_instruction,
            &[
                user_transfer_account.as_account_info(),
                self.pool_account.as_account_info(),
                self.swap_curve_account.as_account_info(),
                self.pool_authority_account.as_account_info(),
                self.token_b_mint_account.as_account_info(),
                self.token_a_mint_account.as_account_info(),
                self.token_b_account.as_account_info(),
                self.token_a_account.as_account_info(),
                self.pool_token_mint_account.as_account_info(),
                self.pool_token_fees_vault_account.as_account_info(),
                token_b_account.as_account_info(),
                token_a_account.as_account_info(),
                self.admin_authority_pool_token_ata.as_account_info(),
                self.pool_token_program_account.as_account_info(),
                self.token_b_program_account.as_account_info(),
                self.token_a_program_account.as_account_info(),
            ],
        )
    }

    pub fn deposit_all_token_types(
        &mut self,
        token_a_account: &mut NativeAccountData,
        token_b_account: &mut NativeAccountData,
        pool_account: &mut NativeAccountData,
        mut instruction: DepositAllTokenTypes,
    ) -> ProgramResult {
        let mut user_transfer_account = NativeAccountData::new(0, system_program::id());
        user_transfer_account.is_signer = true;
        do_process_instruction(
            approve(
                &self.token_a_program_account.key,
                &token_a_account.key,
                &user_transfer_account.key,
                &self.admin_authority.key,
                &[],
                instruction.maximum_token_a_amount,
            )
            .unwrap(),
            &[
                token_a_account.as_account_info(),
                user_transfer_account.as_account_info(),
                self.admin_authority.as_account_info(),
            ],
        )
        .unwrap();

        do_process_instruction(
            approve(
                &self.token_b_program_account.key,
                &token_b_account.key,
                &user_transfer_account.key,
                &self.admin_authority.key,
                &[],
                instruction.maximum_token_b_amount,
            )
            .unwrap(),
            &[
                token_b_account.as_account_info(),
                user_transfer_account.as_account_info(),
                self.admin_authority.as_account_info(),
            ],
        )
        .unwrap();

        // special logic: if we only deposit 1 pool token, we can't withdraw it
        // because we incur a withdrawal fee, so we hack it to not be 1
        if instruction.pool_token_amount == 1 {
            instruction.pool_token_amount = 2;
        }

        let deposit_instruction = ix::deposit_all_token_types(
            &hyperplane::id(),
            &spl_token::id(),
            &spl_token::id(),
            &self.pool_token_program_account.key,
            &self.pool_account.key,
            &self.pool_authority_account.key,
            &user_transfer_account.key,
            &token_a_account.key,
            &token_b_account.key,
            &self.token_a_account.key,
            &self.token_b_account.key,
            &self.pool_token_mint_account.key,
            &pool_account.key,
            &self.token_a_mint_account.key,
            &self.token_b_mint_account.key,
            &self.swap_curve_account.key,
            instruction,
        )
        .unwrap();

        do_process_instruction(
            deposit_instruction,
            &[
                user_transfer_account.as_account_info(),
                self.pool_account.as_account_info(),
                self.swap_curve_account.as_account_info(),
                self.pool_authority_account.as_account_info(),
                self.token_a_mint_account.as_account_info(),
                self.token_b_mint_account.as_account_info(),
                self.token_a_account.as_account_info(),
                self.token_b_account.as_account_info(),
                self.pool_token_mint_account.as_account_info(),
                token_a_account.as_account_info(),
                token_b_account.as_account_info(),
                pool_account.as_account_info(),
                self.pool_token_program_account.as_account_info(),
                self.token_a_program_account.as_account_info(),
                self.token_b_program_account.as_account_info(),
            ],
        )
    }

    pub fn withdraw_all_token_types(
        &mut self,
        pool_account: &mut NativeAccountData,
        token_a_account: &mut NativeAccountData,
        token_b_account: &mut NativeAccountData,
        mut instruction: WithdrawAllTokenTypes,
    ) -> ProgramResult {
        let mut user_transfer_account = NativeAccountData::new(0, system_program::id());
        user_transfer_account.is_signer = true;
        let pool_token_amount = native_token::get_token_balance(pool_account);
        // special logic to avoid withdrawing down to 1 pool token, which
        // eventually causes an error on withdrawing all
        if pool_token_amount.saturating_sub(instruction.pool_token_amount) == 1 {
            instruction.pool_token_amount = pool_token_amount;
        }
        do_process_instruction(
            approve(
                &self.pool_token_program_account.key,
                &pool_account.key,
                &user_transfer_account.key,
                &self.admin_authority.key,
                &[],
                instruction.pool_token_amount,
            )
            .unwrap(),
            &[
                pool_account.as_account_info(),
                user_transfer_account.as_account_info(),
                self.admin_authority.as_account_info(),
            ],
        )
        .unwrap();

        let withdraw_instruction = ix::withdraw_all_token_types(
            &hyperplane::id(),
            &self.pool_token_program_account.key,
            &spl_token::id(),
            &spl_token::id(),
            &self.pool_account.key,
            &self.pool_authority_account.key,
            &user_transfer_account.key,
            &self.pool_token_mint_account.key,
            &self.pool_token_fees_vault_account.key,
            &pool_account.key,
            &self.token_a_account.key,
            &self.token_b_account.key,
            &token_a_account.key,
            &token_b_account.key,
            &self.token_a_mint_account.key,
            &self.token_b_mint_account.key,
            &self.swap_curve_account.key,
            instruction,
        )
        .unwrap();

        do_process_instruction(
            withdraw_instruction,
            &[
                user_transfer_account.as_account_info(),
                self.pool_account.as_account_info(),
                self.swap_curve_account.as_account_info(),
                self.pool_authority_account.as_account_info(),
                self.token_a_mint_account.as_account_info(),
                self.token_b_mint_account.as_account_info(),
                self.token_a_account.as_account_info(),
                self.token_b_account.as_account_info(),
                self.pool_token_mint_account.as_account_info(),
                self.pool_token_fees_vault_account.as_account_info(),
                token_a_account.as_account_info(),
                token_b_account.as_account_info(),
                pool_account.as_account_info(),
                self.pool_token_program_account.as_account_info(),
                self.token_a_program_account.as_account_info(),
                self.token_b_program_account.as_account_info(),
            ],
        )
    }

    pub fn deposit_single_token_type_exact_amount_in(
        &mut self,
        source_token_account: &mut NativeAccountData,
        trade_direction: TradeDirection,
        pool_account: &mut NativeAccountData,
        mut instruction: DepositSingleTokenTypeExactAmountIn,
    ) -> ProgramResult {
        let mut user_transfer_account = NativeAccountData::new(0, system_program::id());
        user_transfer_account.is_signer = true;
        let source_token_program = match trade_direction {
            TradeDirection::AtoB => &mut self.token_a_program_account,
            TradeDirection::BtoA => &mut self.token_b_program_account,
        };
        do_process_instruction(
            approve(
                &source_token_program.key,
                &source_token_account.key,
                &user_transfer_account.key,
                &self.admin_authority.key,
                &[],
                instruction.source_token_amount,
            )
            .unwrap(),
            &[
                source_token_account.as_account_info(),
                user_transfer_account.as_account_info(),
                self.admin_authority.as_account_info(),
            ],
        )
        .unwrap();

        // special logic: if we only deposit 1 pool token, we can't withdraw it
        // because we incur a withdrawal fee, so we hack it to not be 1
        if instruction.minimum_pool_token_amount < 2 {
            instruction.minimum_pool_token_amount = 2;
        }

        let source_token_mint_account = match trade_direction {
            TradeDirection::AtoB => &mut self.token_a_mint_account,
            TradeDirection::BtoA => &mut self.token_b_mint_account,
        };

        let deposit_instruction = ix::deposit_single_token_type(
            &hyperplane::id(),
            &spl_token::id(),
            &spl_token::id(),
            &self.pool_account.key,
            &self.pool_authority_account.key,
            &user_transfer_account.key,
            &source_token_account.key,
            &self.token_a_account.key,
            &self.token_b_account.key,
            &self.pool_token_mint_account.key,
            &pool_account.key,
            &source_token_mint_account.key,
            &self.swap_curve_account.key,
            instruction,
        )
        .unwrap();

        do_process_instruction(
            deposit_instruction,
            &[
                user_transfer_account.as_account_info(),
                self.pool_account.as_account_info(),
                self.swap_curve_account.as_account_info(),
                self.pool_authority_account.as_account_info(),
                source_token_mint_account.as_account_info(),
                self.token_a_account.as_account_info(),
                self.token_b_account.as_account_info(),
                self.pool_token_mint_account.as_account_info(),
                source_token_account.as_account_info(),
                pool_account.as_account_info(),
                self.pool_token_program_account.as_account_info(),
                self.token_a_program_account.as_account_info(),
            ],
        )
    }

    pub fn withdraw_single_token_type_exact_amount_out(
        &mut self,
        pool_account: &mut NativeAccountData,
        trade_direction: TradeDirection,
        destination_token_account: &mut NativeAccountData,
        mut instruction: WithdrawSingleTokenTypeExactAmountOut,
    ) -> ProgramResult {
        let mut user_transfer_account = NativeAccountData::new(0, system_program::id());
        user_transfer_account.is_signer = true;
        let pool_token_amount = native_token::get_token_balance(pool_account);
        // special logic to avoid withdrawing down to 1 pool token, which
        // eventually causes an error on withdrawing all
        if pool_token_amount.saturating_sub(instruction.maximum_pool_token_amount) == 1 {
            instruction.maximum_pool_token_amount = pool_token_amount;
        }
        do_process_instruction(
            approve(
                &self.pool_token_program_account.key,
                &pool_account.key,
                &user_transfer_account.key,
                &self.admin_authority.key,
                &[],
                instruction.maximum_pool_token_amount,
            )
            .unwrap(),
            &[
                pool_account.as_account_info(),
                user_transfer_account.as_account_info(),
                self.admin_authority.as_account_info(),
            ],
        )
        .unwrap();

        let destination_token_program = match trade_direction {
            TradeDirection::AtoB => &mut self.token_a_program_account,
            TradeDirection::BtoA => &mut self.token_b_program_account,
        };
        let destination_token_mint_account = match trade_direction {
            TradeDirection::AtoB => &mut self.token_a_mint_account,
            TradeDirection::BtoA => &mut self.token_b_mint_account,
        };
        let withdraw_instruction = ix::withdraw_single_token_type_exact_amount_out(
            &hyperplane::id(),
            &spl_token::id(),
            &spl_token::id(),
            &self.pool_account.key,
            &self.pool_authority_account.key,
            &user_transfer_account.key,
            &self.pool_token_mint_account.key,
            &self.pool_token_fees_vault_account.key,
            &pool_account.key,
            &self.token_a_account.key,
            &self.token_b_account.key,
            &destination_token_account.key,
            &destination_token_mint_account.key,
            &self.swap_curve_account.key,
            instruction,
        )
        .unwrap();

        do_process_instruction(
            withdraw_instruction,
            &[
                user_transfer_account.as_account_info(),
                self.pool_account.as_account_info(),
                self.swap_curve_account.as_account_info(),
                self.pool_authority_account.as_account_info(),
                destination_token_mint_account.as_account_info(),
                self.token_a_account.as_account_info(),
                self.token_b_account.as_account_info(),
                self.pool_token_mint_account.as_account_info(),
                self.pool_token_fees_vault_account.as_account_info(),
                destination_token_account.as_account_info(),
                pool_account.as_account_info(),
                self.pool_token_program_account.as_account_info(),
                destination_token_program.as_account_info(),
            ],
        )
    }

    pub fn withdraw_all(
        &mut self,
        pool_account: &mut NativeAccountData,
        token_a_account: &mut NativeAccountData,
        token_b_account: &mut NativeAccountData,
    ) -> ProgramResult {
        let pool_token_amount = native_token::get_token_balance(pool_account);
        if pool_token_amount > 0 {
            let instruction = WithdrawAllTokenTypes {
                pool_token_amount,
                minimum_token_a_amount: 0,
                minimum_token_b_amount: 0,
            };
            self.withdraw_all_token_types(
                pool_account,
                token_a_account,
                token_b_account,
                instruction,
            )
        } else {
            Ok(())
        }
    }
}
