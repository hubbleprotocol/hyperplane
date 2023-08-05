use anchor_lang::{
    error,
    prelude::{msg, AccountInfo, Clock, Result, SolanaSysvar},
    solana_program::clock::Epoch,
};
use anchor_spl::token_interface::spl_token_2022::extension::{
    transfer_fee::TransferFeeConfig, BaseStateWithExtensions, StateWithExtensions,
};

use crate::{curve::fees::Fees, error::SwapError, to_u64, try_math, utils::math::TryMath};

/// Subtract token mint transfer fees for actual amount received by the user post-transfer fees
pub fn sub_transfer_fee(
    transfer_fees: Option<(&TransferFeeConfig, Epoch)>,
    amount: u128,
) -> Result<u128> {
    let source_amt_sub_xfer_fees = match transfer_fees {
        None => amount,
        Some((xfer_fee_config, epoch)) => {
            let xfer_fee = xfer_fee_config
                .calculate_epoch_fee(epoch, to_u64!(amount)?)
                .ok_or_else(|| error!(SwapError::FeeCalculationFailure))?;
            let amount_sub_fee = try_math!(amount.try_sub(xfer_fee.into()))?;
            msg!(
                "Subtract token transfer fee: fee={}, amount={}, amount_sub_fee={}",
                xfer_fee,
                amount,
                amount_sub_fee
            );
            amount_sub_fee
        }
    };
    Ok(source_amt_sub_xfer_fees)
}

/// Subtract token mint transfer fees for actual amount received by the user post-transfer fees
pub fn sub_transfer_fee2(mint_acc_info: &AccountInfo, amount: u64) -> Result<u64> {
    let mint_data = mint_acc_info.data.borrow();
    let mint = StateWithExtensions::<anchor_spl::token_2022::spl_token_2022::state::Mint>::unpack(
        &mint_data,
    )?;
    let amount = if let Ok(transfer_fee_config) = mint.get_extension::<TransferFeeConfig>() {
        let transfer_fee = transfer_fee_config
            .calculate_epoch_fee(Clock::get()?.epoch, amount)
            .ok_or_else(|| error!(SwapError::FeeCalculationFailure))?;
        let amount_sub_fee = try_math!(amount.try_sub(transfer_fee))?;
        msg!(
            "Subtract token transfer fee: fee={}, amount={}, amount_sub_fee={}",
            transfer_fee,
            amount,
            amount_sub_fee
        );
        amount_sub_fee
    } else {
        amount
    };
    Ok(amount)
}

/// Subtract token mint transfer fees for actual amount received by the pool post-transfer fees
///
/// There are potentially 3 input transfers:
/// 1. User -> Pool
/// 2. User -> Fees
/// 3. User -> Host Fees (optional)
pub fn sub_input_transfer_fees(
    mint_acc_info: &AccountInfo,
    fees: &Fees,
    amount_in: u64,
    host_fee: bool,
) -> Result<u64> {
    let mint_data = mint_acc_info.data.borrow();
    let mint = StateWithExtensions::<anchor_spl::token_2022::spl_token_2022::state::Mint>::unpack(
        &mint_data,
    )?;
    let amount = if let Ok(transfer_fee_config) = mint.get_extension::<TransferFeeConfig>() {
        let owner_and_host_fee = fees.owner_trading_fee(amount_in.into())?;
        let epoch = Clock::get()?.epoch;
        let (host_fee, host_transfer_fee) = if host_fee {
            let host_fee = fees.host_fee(owner_and_host_fee)?;
            (
                host_fee,
                transfer_fee_config
                    .calculate_epoch_fee(epoch, to_u64!(host_fee)?)
                    .ok_or_else(|| error!(SwapError::FeeCalculationFailure))?,
            )
        } else {
            (0, 0)
        };
        let owner_fee = try_math!(owner_and_host_fee.try_sub(host_fee))?;
        let owner_transfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, to_u64!(owner_fee)?)
            .ok_or_else(|| error!(SwapError::FeeCalculationFailure))?;

        let vault_amount_in = try_math!(amount_in.try_sub(to_u64!(owner_and_host_fee)?))?;
        let vault_transfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, vault_amount_in)
            .ok_or_else(|| error!(SwapError::FeeCalculationFailure))?;

        let amount_sub_fees = try_math!(try_math!(try_math!(
            amount_in.try_sub(vault_transfer_fee)
        )?
        .try_sub(owner_transfer_fee))?
        .try_sub(host_transfer_fee))?;

        msg!(
                "Subtract input token transfer fee: vault_transfer_amount={}, vault_transfer_fee={}, owner_fee={}, owner_fee_transfer_fee={}, host_fee={}, host_fee_transfer_fee={} amount={}, input_amount_sub_transfer_fees={}",
                vault_amount_in,
                vault_transfer_fee,
                owner_fee,
                owner_transfer_fee,
                host_fee,
                host_transfer_fee,
                amount_in,
                amount_sub_fees
            );
        amount_sub_fees
    } else {
        amount_in
    };
    Ok(amount)
}

/// Get transfer fee config and epoch if present on token 2022 mint
pub fn get_transfer_fee_config<'mint>(
    mint: &'mint StateWithExtensions<anchor_spl::token_2022::spl_token_2022::state::Mint>,
) -> Option<(&'mint TransferFeeConfig, Epoch)> {
    let config = if let Ok(transfer_fee_config) = mint.get_extension::<TransferFeeConfig>() {
        Some((transfer_fee_config, Clock::get().unwrap().epoch))
    } else {
        None
    };
    config
}

/// Add token mint transfer fees for actual amount sent pre-transfer fees
pub fn add_inverse_transfer_fee(
    transfer_fees: Option<(&TransferFeeConfig, Epoch)>,
    post_fee_amount: u128,
) -> Result<u128> {
    let amount = match transfer_fees {
        None => post_fee_amount,
        Some((xfer_fee_config, epoch)) => {
            let xfer_fee = xfer_fee_config
                .calculate_inverse_epoch_fee(epoch, to_u64!(post_fee_amount)?)
                .ok_or_else(|| error!(SwapError::FeeCalculationFailure))?;
            let amount_add_fee = try_math!(post_fee_amount.try_add(xfer_fee.into()))?;
            msg!(
                "Add token transfer fee: fee={}, amount={}, amount_add_fee={}",
                xfer_fee,
                post_fee_amount,
                amount_add_fee
            );
            amount_add_fee
        }
    };
    Ok(amount)
}

/// Add token mint transfer fees for actual amount sent pre-transfer fees
pub fn add_inverse_transfer_fee2(mint_acc_info: &AccountInfo, post_fee_amount: u64) -> Result<u64> {
    let mint_data = mint_acc_info.data.borrow();
    let mint = StateWithExtensions::<anchor_spl::token_2022::spl_token_2022::state::Mint>::unpack(
        &mint_data,
    )?;
    let amount = if let Ok(transfer_fee_config) = mint.get_extension::<TransferFeeConfig>() {
        let transfer_fee = transfer_fee_config
            .calculate_inverse_epoch_fee(Clock::get()?.epoch, post_fee_amount)
            .ok_or_else(|| error!(SwapError::FeeCalculationFailure))?;
        let amount_add_fee = try_math!(post_fee_amount.try_add(transfer_fee))?;
        msg!(
            "Add token transfer fee: fee={}, amount={}, amount_add_fee={}",
            transfer_fee,
            post_fee_amount,
            amount_add_fee
        );
        amount_add_fee
    } else {
        post_fee_amount
    };
    Ok(amount)
}

pub fn round_transfer_fees_if_needed(
    transfer_fees: Option<(&TransferFeeConfig, Epoch)>,
    amount: u128,
) -> Result<u128> {
    let amount_sub_fee = sub_transfer_fee(transfer_fees, amount)?;
    let amount = if amount_sub_fee == 0 {
        add_inverse_transfer_fee(transfer_fees, 1)?
    } else {
        amount_sub_fee
    };
    Ok(amount)
}

#[cfg(test)]
mod test {
    use anchor_lang::solana_program::{clock::Epoch, program_option::COption, pubkey::Pubkey};
    use anchor_spl::token_2022::{
        spl_token_2022,
        spl_token_2022::{
            extension::{transfer_fee::TransferFee, ExtensionType, StateWithExtensionsMut},
            pod::OptionalNonZeroPubkey,
        },
    };
    use proptest::{prop_assume, proptest};

    use super::*;
    use crate::instructions::test::runner::syscall_stubs::test_syscall_stubs;

    #[test]
    pub fn test_sub_transfer_fee_when_no_transfer_fees() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 0);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amount = sub_transfer_fee2(&mint_info, 10_000).unwrap();

        assert_eq!(amount, 10_000);
    }

    #[test]
    pub fn test_sub_transfer_fee_when_10_bps_transfer_fee() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amount = sub_transfer_fee2(&mint_info, 10_000).unwrap();

        assert_eq!(amount, 9990);
    }

    #[test]
    pub fn test_sub_transfer_fee_rounds_up_when_small_fee() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amount = sub_transfer_fee2(&mint_info, 100).unwrap();

        assert_eq!(amount, 99);
    }

    #[test]
    pub fn test_add_inverse_transfer_fee_when_no_transfer_fees() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 0);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amount = add_inverse_transfer_fee2(&mint_info, 10_000).unwrap();

        assert_eq!(amount, 10_000);
    }

    #[test]
    pub fn test_add_inverse_transfer_fee_when_10_bps_transfer_fee() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amount = add_inverse_transfer_fee2(&mint_info, 9990).unwrap();

        assert_eq!(amount, 10_000);
    }

    #[test]
    pub fn test_add_inverse_transfer_fee_rounds_up_when_small_fee() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amount = add_inverse_transfer_fee2(&mint_info, 100).unwrap();

        assert_eq!(amount, 101);
    }

    #[test]
    pub fn test_sub_then_add_inverse_transfer_fee_when_10_bps_transfer_fee() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let receive_amount = sub_transfer_fee2(&mint_info, 10_000_000).unwrap();
        let original = add_inverse_transfer_fee2(&mint_info, receive_amount).unwrap();

        assert_eq!(original, 10_000_000);
    }

    #[test]
    pub fn test_sub_input_transfer_fee_when_no_transfer_fees_or_protocol_fees() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 0);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amount = sub_input_transfer_fees(&mint_info, &Fees::default(), 10_000, false).unwrap();

        assert_eq!(amount, 10_000);
    }

    #[test]
    pub fn test_sub_input_transfer_fee_when_10bps_transfer_fees_and_no_protocol_fees() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amount = sub_input_transfer_fees(&mint_info, &Fees::default(), 10_000, false).unwrap();

        // 1 transfer fee of 10 bps
        assert_eq!(amount, 9990);
    }

    #[test]
    pub fn test_sub_input_transfer_fee_when_10bps_transfer_fees_and_owner_protocol_fees() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let fees = Fees {
            owner_trade_fee_numerator: 10,
            owner_trade_fee_denominator: 10_000,
            ..Default::default()
        };

        let amount = sub_input_transfer_fees(&mint_info, &fees, 10_000_000, false).unwrap();

        // Raw owner fee amount is 10_000 (10 bps of 10M)
        // Raw owner transfer fee is 10 (10 bps of 10_000)
        // Vault transfer amount is 9_990_000 (10M - 10_000)
        // not -10 because we re-take the owner fee from the total amount - all transfer fees
        // so the proportion of the owner fee is the same
        // vault transfer fee is 9990 (10 bps of 9_990_000)
        // 2 transfer fees equal to 10_000 total (9900 + 10)
        assert_eq!(amount, 9_990_000);
    }

    #[test]
    pub fn test_sub_input_transfer_fee_when_10bps_transfer_fees_and_owner_and_host_protocol_fees() {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let fees = Fees {
            owner_trade_fee_numerator: 10,
            owner_trade_fee_denominator: 10_000,
            host_fee_numerator: 10,
            host_fee_denominator: 10_000,
            ..Default::default()
        };

        let amount = sub_input_transfer_fees(&mint_info, &fees, 100_000_000_000_000, true).unwrap();

        // Owner fee amount is 100_000_000_000 (10 bps of 100_000B)
        // Host fee 10_000_000 (10 bps of 100_000_000_000) taken from the owner fee which is now 99_990_000_000 (100_000_000_000 - 10_000_000)
        // Owner transfer fee is 99_990_000 (10 bps of 99_990_000_000)
        // Host transfer fee is 10_000 (10 bps of 10_000_000)
        // Vault transfer amount is 99_900_000_000_000 (100_000B - 100_000_000_000)
        // vault transfer fee is 99_900_000_000 (10 bps of 99_900_000_000_000)
        // 3 transfer fees equal to 100_000_000_000 total (99_900_000_000 + 99_990_000 + 10_000)
        assert_eq!(amount, 99_900_000_000_000);
    }

    #[test]
    pub fn test_sub_input_transfer_fee_when_10bps_transfer_fees_and_owner_and_small_host_protocol_fees(
    ) {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let fees = Fees {
            owner_trade_fee_numerator: 10,
            owner_trade_fee_denominator: 10_000,
            host_fee_numerator: 10,
            host_fee_denominator: 10_000,
            ..Default::default()
        };

        let amount = sub_input_transfer_fees(&mint_info, &fees, 100_000_000, true).unwrap();

        // Owner fee amount is 100_000 (10 bps of 100M)
        // Host fee 100 (10 bps of 100_000) taken from the owner fee which is now 99_900 (100_000 - 100)
        // Owner transfer fee is 100 (10 bps of 100_000)
        // Host transfer fee is 1 (10 bps of 100 rounded up)
        // Vault transfer amount is 99_900_000 (100M - 100_000)
        // Vault transfer fee is 99_900 (10 bps of 99_900_000)
        // 3 transfer fees equal to 100_001 total (99_900 + 100 + 1)
        assert_eq!(amount, 99_899_999);
    }

    #[test]
    pub fn test_sub_input_transfer_fee_when_10bps_transfer_fees_and_both_owner_and_host_protocol_fees_small(
    ) {
        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, 10);

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let fees = Fees {
            owner_trade_fee_numerator: 10,
            owner_trade_fee_denominator: 10_000,
            host_fee_numerator: 10,
            host_fee_denominator: 10_000,
            ..Default::default()
        };

        let amount = sub_input_transfer_fees(&mint_info, &fees, 10_000_000, true).unwrap();

        // Owner fee amount is 10_000 (10 bps of 10M)
        // Host fee 10 (10 bps of 10_000) taken from the owner fee which is now 9_990 (10_000 - 10)
        // Owner transfer fee is 10 (10 bps of 9_990 rounded up)
        // Host transfer fee is 1 (10 bps of 9 rounded up)
        // Vault transfer amount is 9_990_000 (10M - 10_000)
        // Vault transfer fee is 9990 (10 bps of 9_990_000)
        // 3 transfer fees equal to 10_001 total (9990 + 10 + 1)
        assert_eq!(amount, 9_989_999);
    }

    proptest! {
        #[test]
        fn test_sub_then_add_inverse_transfer_fee_should_be_same_or_one_less(
            amount in 1..u32::MAX as u64,
            transfer_fee_bps in 0..10_000_u64,
        ) {
            test_syscall_stubs();

            let mut mint_data = mint_with_fee_data();
            mint_with_transfer_fee(&mut mint_data, 10);

            let key = Pubkey::new_unique();
            let mut lamports = u64::MAX;
            let token_program = spl_token_2022::id();
            let mint_info = AccountInfo::new(
                &key,
                false,
                false,
                &mut lamports,
                &mut mint_data,
                &token_program,
                false,
                Epoch::default(),
            );

            let receive_amount = sub_transfer_fee2(&mint_info, amount).unwrap();
            let original = add_inverse_transfer_fee2(&mint_info, receive_amount).unwrap();

            assert!(amount - original <= 1, "original: {}, amount: {}, diff: {}, transfer_fee_bps: {}, receive_amount={}", original, amount, amount - original, transfer_fee_bps, receive_amount);
        }
    }

    proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 10000, max_global_rejects: u32::MAX, .. proptest::prelude::ProptestConfig::default()
        })]
        #[test]
        fn test_sub_input_fees_same_or_less_after_re_adding(
            amount in 1..100000 as u64,
            owner_trade_fee_numerator in 0..100_000_u64,
            owner_trade_fee_denominator in 1..100_000_u64,
            host_fee_numerator in 0..100_000_u64,
            host_fee_denominator in 1..100_000_u64,
            transfer_fee_bps in 0..1000_u64,
            host_fees: bool,
        ) {
            let host_fees = false;
            prop_assume!(host_fee_numerator <= host_fee_denominator);
            prop_assume!(owner_trade_fee_numerator <= owner_trade_fee_denominator);
            test_syscall_stubs();

            let mut mint_data = mint_with_fee_data();
            mint_with_transfer_fee(&mut mint_data, u16::try_from(transfer_fee_bps).unwrap());

            let key = Pubkey::new_unique();
            let mut lamports = u64::MAX;
            let token_program = spl_token_2022::id();
            let mint_info = AccountInfo::new(
                &key,
                false,
                false,
                &mut lamports,
                &mut mint_data,
                &token_program,
                false,
                Epoch::default(),
            );

            let fees = Fees {
                owner_trade_fee_numerator,
                owner_trade_fee_denominator,
                host_fee_numerator,
                host_fee_denominator,
                ..Default::default()
            };

            let amount_sub_fees = sub_input_transfer_fees(&mint_info, &fees, amount, host_fees).unwrap();

            let estimated_transfer_fees = amount - amount_sub_fees;

            let owner_and_host_fee = fees.owner_trading_fee(amount_sub_fees.into()).unwrap();
            let host_fee_sub_fees = if host_fees {
                fees.host_fee(owner_and_host_fee).unwrap() as u64
            } else {
                0
            };

            let owner_fee_sub_fees = (owner_and_host_fee as u64).checked_sub(host_fee_sub_fees).unwrap();
            let vault_amount_sub_fees = amount_sub_fees.checked_sub(owner_and_host_fee as u64).unwrap();

            assert_eq!(amount_sub_fees, vault_amount_sub_fees + owner_fee_sub_fees + host_fee_sub_fees, "amount: {}, vault_amount: {}, host_and_owner_fee: {}, owner_fee: {}, host_fee: {}, amount_sub_fees: {}", amount, vault_amount_sub_fees, owner_and_host_fee, owner_fee_sub_fees, host_fee_sub_fees, amount_sub_fees);

            let vault_amount_add_fees = add_inverse_transfer_fee2(&mint_info, vault_amount_sub_fees).unwrap();
            let owner_amount_add_fees = add_inverse_transfer_fee2(&mint_info, owner_fee_sub_fees).unwrap();
            let host_amount_add_fees = if host_fees {
                add_inverse_transfer_fee2(&mint_info, host_fee_sub_fees).unwrap()
            } else {
                0
            };

            let actual_vault_transfer_fee = vault_amount_add_fees - vault_amount_sub_fees;
            let actual_owner_transfer_fee = owner_amount_add_fees - owner_fee_sub_fees;
            let actual_host_transfer_fee = host_amount_add_fees - host_fee_sub_fees;
            let actual_transfer_fees = actual_vault_transfer_fee + actual_owner_transfer_fee + actual_host_transfer_fee;

            if host_fees {
                let amount_with_fees = vault_amount_add_fees + owner_amount_add_fees + host_amount_add_fees;
                let msg = format!("\namount={amount}\namount_with_xfer_fees={amount_with_fees}\ntransfer_fee_bps={transfer_fee_bps}\nestimated_transfer_fees={estimated_transfer_fees}\nactual_transfer_fees={actual_transfer_fees}\nvault_amount={vault_amount_sub_fees}\n\tvault_amount_xfer_fees={actual_vault_transfer_fee}\n\tvault_amount_add_xfer_fees={vault_amount_add_fees}\nowner_fee_amount={owner_fee_sub_fees}\n\towner_xfer_fees={actual_owner_transfer_fee}\n\towner_amount_add_xfer_fees={owner_amount_add_fees}\nhost_fee_amount={host_fee_sub_fees}\n\thost_amount_xfer_fees={actual_host_transfer_fee}\n\thost_amount_add_xfer_fees={host_amount_add_fees}\nhost_and_owner_fee={owner_and_host_fee}\namount_sub_xfer_fees={amount_sub_fees}\n");
                assert!(amount_with_fees <= amount, "{}", msg);
                let diff = amount - amount_with_fees;
                assert!(diff <= 3, "\ndiff={}{}", diff, msg);
            } else {
                let amount_with_fees = vault_amount_add_fees + owner_amount_add_fees;
                let msg = format!("\namount={amount}\namount_with_xfer_fees={amount_with_fees}\ntransfer_fee_bps={transfer_fee_bps}\nestimated_transfer_fees={estimated_transfer_fees}\nactual_transfer_fees={actual_transfer_fees}\nvault_amount={vault_amount_sub_fees}\n\tvault_amount_xfer_fees={actual_vault_transfer_fee}\n\tvault_amount_add_xfer_fees={vault_amount_add_fees}\nowner_fee_amount={owner_fee_sub_fees}\n\towner_xfer_fees={actual_owner_transfer_fee}\n\towner_amount_add_xfer_fees={owner_amount_add_fees}\namount_sub_xfer_fees={amount_sub_fees}\n");
                assert!(amount_with_fees <= amount, "{}", msg);
                let diff = amount - amount_with_fees;
                assert!(diff <= 2, "\ndiff={}{}", diff, msg);
            }
        }
    }

    #[test]
    fn man_test() {
        let amount = 2006024888;
        let owner_trade_fee_numerator = 2933;
        let owner_trade_fee_denominator = 14681;
        let host_fee_numerator = 58494;
        let host_fee_denominator = 69432;
        let transfer_fee_bps = 226;
        let host_fees = true;

        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, u16::try_from(transfer_fee_bps).unwrap());

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let fees = Fees {
            owner_trade_fee_numerator,
            owner_trade_fee_denominator,
            host_fee_numerator,
            host_fee_denominator,
            ..Default::default()
        };

        let amount_sub_xfer_fees =
            sub_input_transfer_fees(&mint_info, &fees, amount, host_fees).unwrap();

        let estimated_transfer_fees = amount - amount_sub_xfer_fees;

        let owner_and_host_fee_sub_xfer_fees =
            fees.owner_trading_fee(amount_sub_xfer_fees.into()).unwrap();
        let host_fee_sub_xfer_fees = if host_fees {
            fees.host_fee(owner_and_host_fee_sub_xfer_fees).unwrap() as u64
        } else {
            0
        };

        let owner_fee_sub_xfer_fees =
            (owner_and_host_fee_sub_xfer_fees as u64).saturating_sub(host_fee_sub_xfer_fees);
        let vault_amount_sub_xfer_fees =
            amount_sub_xfer_fees.saturating_sub(owner_and_host_fee_sub_xfer_fees as u64);

        assert_eq!(amount_sub_xfer_fees, vault_amount_sub_xfer_fees + owner_fee_sub_xfer_fees + host_fee_sub_xfer_fees, "amount: {}, vault_amount: {}, host_and_owner_fee: {}, owner_fee: {}, host_fee: {}, amount_sub_fees: {}", amount, vault_amount_sub_xfer_fees, owner_and_host_fee_sub_xfer_fees, owner_fee_sub_xfer_fees, host_fee_sub_xfer_fees, amount_sub_xfer_fees);

        let vault_amount_add_fees =
            add_inverse_transfer_fee2(&mint_info, vault_amount_sub_xfer_fees).unwrap();
        let owner_amount_add_fees =
            add_inverse_transfer_fee2(&mint_info, owner_fee_sub_xfer_fees).unwrap();
        let host_amount_add_fees = if host_fees {
            add_inverse_transfer_fee2(&mint_info, host_fee_sub_xfer_fees).unwrap()
        } else {
            0
        };

        let actual_vault_transfer_fee = vault_amount_add_fees - vault_amount_sub_xfer_fees;
        let actual_owner_transfer_fee = owner_amount_add_fees - owner_fee_sub_xfer_fees;
        let actual_host_transfer_fee = host_amount_add_fees - host_fee_sub_xfer_fees;
        let actual_transfer_fees =
            actual_vault_transfer_fee + actual_owner_transfer_fee + actual_host_transfer_fee;

        let amount_with_fees = vault_amount_add_fees + owner_amount_add_fees + host_amount_add_fees;
        let msg = format!("\namount={amount}\namount_with_xfer_fees={amount_with_fees}\ntransfer_fee_bps={transfer_fee_bps}\nestimated_transfer_fees={estimated_transfer_fees}\nactual_transfer_fees={actual_transfer_fees}\nvault_amount={vault_amount_sub_xfer_fees}\n\tvault_amount_xfer_fees={actual_vault_transfer_fee}\n\tvault_amount_add_xfer_fees={vault_amount_add_fees}\nowner_fee_amount={owner_fee_sub_xfer_fees}\n\towner_xfer_fees={actual_owner_transfer_fee}\n\towner_amount_add_xfer_fees={owner_amount_add_fees}\nhost_fee_amount={host_fee_sub_xfer_fees}\n\thost_amount_xfer_fees={actual_host_transfer_fee}\n\thost_amount_add_xfer_fees={host_amount_add_fees}\nhost_and_owner_fee={owner_and_host_fee_sub_xfer_fees}\namount_sub_xfer_fees={amount_sub_xfer_fees}\n");
        assert!(amount_with_fees <= amount, "{}", msg);
        let diff = amount - amount_with_fees;
        assert!(diff <= 3, "\ndiff={}{}", diff, msg);
    }

    #[test]
    fn man_test1111() {
        let amount = 23210;
        let owner_trade_fee_numerator = 3819;
        let owner_trade_fee_denominator = 31977;
        let host_fee_numerator = 0;
        let host_fee_denominator = 1;
        let transfer_fee_bps = 981;
        let host_fees = false;

        test_syscall_stubs();

        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, u16::try_from(transfer_fee_bps).unwrap());

        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let fees = Fees {
            owner_trade_fee_numerator,
            owner_trade_fee_denominator,
            host_fee_numerator,
            host_fee_denominator,
            ..Default::default()
        };

        let amount_sub_xfer_fees =
            sub_input_transfer_fees(&mint_info, &fees, amount, host_fees).unwrap();

        let estimated_transfer_fees = amount - amount_sub_xfer_fees;

        let owner_and_host_fee_sub_xfer_fees =
            fees.owner_trading_fee(amount_sub_xfer_fees.into()).unwrap();
        let host_fee_sub_xfer_fees = if host_fees {
            fees.host_fee(owner_and_host_fee_sub_xfer_fees).unwrap() as u64
        } else {
            0
        };

        let owner_fee_sub_xfer_fees =
            (owner_and_host_fee_sub_xfer_fees as u64).saturating_sub(host_fee_sub_xfer_fees);
        let vault_amount_sub_xfer_fees =
            amount_sub_xfer_fees.saturating_sub(owner_and_host_fee_sub_xfer_fees as u64);

        assert_eq!(amount_sub_xfer_fees, vault_amount_sub_xfer_fees + owner_fee_sub_xfer_fees + host_fee_sub_xfer_fees, "amount: {}, vault_amount: {}, host_and_owner_fee: {}, owner_fee: {}, host_fee: {}, amount_sub_fees: {}", amount, vault_amount_sub_xfer_fees, owner_and_host_fee_sub_xfer_fees, owner_fee_sub_xfer_fees, host_fee_sub_xfer_fees, amount_sub_xfer_fees);

        let vault_amount_add_fees =
            add_inverse_transfer_fee2(&mint_info, vault_amount_sub_xfer_fees).unwrap();
        let owner_amount_add_fees =
            add_inverse_transfer_fee2(&mint_info, owner_fee_sub_xfer_fees).unwrap();
        let host_amount_add_fees = if host_fees {
            add_inverse_transfer_fee2(&mint_info, host_fee_sub_xfer_fees).unwrap()
        } else {
            0
        };

        let actual_vault_transfer_fee = vault_amount_add_fees - vault_amount_sub_xfer_fees;
        let actual_owner_transfer_fee = owner_amount_add_fees - owner_fee_sub_xfer_fees;
        let actual_host_transfer_fee = host_amount_add_fees - host_fee_sub_xfer_fees;
        let actual_transfer_fees =
            actual_vault_transfer_fee + actual_owner_transfer_fee + actual_host_transfer_fee;

        let amount_with_fees = vault_amount_add_fees + owner_amount_add_fees + host_amount_add_fees;
        let msg = format!("\namount={amount}\namount_with_xfer_fees={amount_with_fees}\ntransfer_fee_bps={transfer_fee_bps}\nestimated_transfer_fees={estimated_transfer_fees}\nactual_transfer_fees={actual_transfer_fees}\nvault_amount={vault_amount_sub_xfer_fees}\n\tvault_amount_xfer_fees={actual_vault_transfer_fee}\n\tvault_amount_add_xfer_fees={vault_amount_add_fees}\nowner_fee_amount={owner_fee_sub_xfer_fees}\n\towner_xfer_fees={actual_owner_transfer_fee}\n\towner_amount_add_xfer_fees={owner_amount_add_fees}\nhost_fee_amount={host_fee_sub_xfer_fees}\n\thost_amount_xfer_fees={actual_host_transfer_fee}\n\thost_amount_add_xfer_fees={host_amount_add_fees}\nhost_and_owner_fee={owner_and_host_fee_sub_xfer_fees}\namount_sub_xfer_fees={amount_sub_xfer_fees}\n");
        assert!(amount_with_fees <= amount, "{}", msg);
        let diff = amount - amount_with_fees;
        assert!(diff <= 3, "\ndiff={}{}", diff, msg);
    }

    #[test]
    fn man_test_2() {
        let owner_trade_fee_numerator = 14842;
        let owner_trade_fee_denominator = 52976;
        let host_fee_numerator = 43369;
        let host_fee_denominator = 89689;
        let fees = Fees {
            owner_trade_fee_numerator,
            owner_trade_fee_denominator,
            host_fee_numerator,
            host_fee_denominator,
            ..Default::default()
        };

        test_syscall_stubs();
        let transfer_fee_bps = 394;
        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, u16::try_from(transfer_fee_bps).unwrap());
        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let epoch = Clock::get().unwrap().epoch;
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let host_fee = 171603172_u64;

        let amount_sub_xfer_fees = 1216786566_u64;
        let owner_and_host_fee_sub_xfer_fees =
            fees.owner_trading_fee(amount_sub_xfer_fees.into()).unwrap();
        let host_fee_sub_xfer_fees =
            fees.host_fee(owner_and_host_fee_sub_xfer_fees).unwrap() as u64;

        let mint_data = mint_info.data.borrow();
        let mint =
            StateWithExtensions::<anchor_spl::token_2022::spl_token_2022::state::Mint>::unpack(
                &mint_data,
            )
            .unwrap();
        let transfer_fee_config = mint.get_extension::<TransferFeeConfig>().unwrap();
        let host_fee_xfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, to_u64!(host_fee).unwrap())
            .unwrap();
        let host_fee_sub_xfer_fee = host_fee.saturating_sub(host_fee_xfer_fee);

        let host_fee_readd_xfer_fee =
            add_inverse_transfer_fee2(&mint_info, host_fee_sub_xfer_fee).unwrap();

        assert!(
            host_fee_readd_xfer_fee <= host_fee,
            "host_fee_readd_xfer_fee: {}, host_fee: {}",
            host_fee_readd_xfer_fee,
            host_fee
        );
    }

    #[test]
    fn man_test_3() {
        let amount = 2006024888_u64;
        let owner_trade_fee_numerator = 2933;
        let owner_trade_fee_denominator = 14681;
        let host_fee_numerator = 58494;
        let host_fee_denominator = 69432;
        let transfer_fee_bps = 226;
        let host_fees = true;

        let fees = Fees {
            owner_trade_fee_numerator,
            owner_trade_fee_denominator,
            host_fee_numerator,
            host_fee_denominator,
            ..Default::default()
        };

        test_syscall_stubs();
        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, u16::try_from(transfer_fee_bps).unwrap());
        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let epoch = Clock::get().unwrap().epoch;
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let owner_and_host_fee = fees.owner_trading_fee(amount.into()).unwrap();
        let host_fee = fees.host_fee(owner_and_host_fee).unwrap() as u64;
        let owner_fee = owner_and_host_fee.saturating_sub(host_fee as u128) as u64;
        let vault_amount = amount.saturating_sub(owner_and_host_fee as u64) as u64;

        assert_eq!(amount, owner_fee + host_fee + vault_amount);
        let mint_data = mint_info.data.borrow();
        let mint =
            StateWithExtensions::<anchor_spl::token_2022::spl_token_2022::state::Mint>::unpack(
                &mint_data,
            )
            .unwrap();
        let transfer_fee_config = mint.get_extension::<TransferFeeConfig>().unwrap();
        let host_fee_xfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, to_u64!(host_fee).unwrap())
            .unwrap();
        let owner_fee_xfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, to_u64!(owner_fee).unwrap())
            .unwrap();
        let vault_amount_xfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, to_u64!(vault_amount).unwrap())
            .unwrap();

        let host_feeee = host_fee.saturating_sub(host_fee_xfer_fee);

        let amount_re = amount
            .saturating_sub(host_fee_xfer_fee)
            .saturating_sub(owner_fee_xfer_fee)
            .saturating_sub(vault_amount_xfer_fee);
        let owner_and_host_fee_re = fees.owner_trading_fee(amount_re.into()).unwrap();
        let host_fee_re = fees.host_fee(owner_and_host_fee_re).unwrap() as u64;
        let owner_fee_re = owner_and_host_fee_re.saturating_sub(host_fee_re as u128) as u64;
        let vault_amount_re = amount_re.saturating_sub(owner_and_host_fee_re as u64) as u64;

        let host_fee_readd_xfer_fee = add_inverse_transfer_fee2(&mint_info, host_fee_re).unwrap();

        assert_eq!(host_fee, host_fee_readd_xfer_fee);

        let host_fee_sub_xfer_fee = host_fee.saturating_sub(host_fee_xfer_fee);

        let host_fee_readd_xfer_fee =
            add_inverse_transfer_fee2(&mint_info, host_fee_sub_xfer_fee).unwrap();

        assert!(
            host_fee_readd_xfer_fee <= host_fee,
            "host_fee_readd_xfer_fee: {}, host_fee: {}",
            host_fee_readd_xfer_fee,
            host_fee
        );
    }

    // #[test]
    fn man_test_4() {
        let owner_trade_fee_numerator = 40;
        let owner_trade_fee_denominator = 100;
        let host_fee_numerator = 125;
        let host_fee_denominator = 1000;
        let fees = Fees {
            owner_trade_fee_numerator,
            owner_trade_fee_denominator,
            host_fee_numerator,
            host_fee_denominator,
            ..Default::default()
        };

        test_syscall_stubs();
        let transfer_fee_bps = 1_000;
        let mut mint_data = mint_with_fee_data();
        mint_with_transfer_fee(&mut mint_data, u16::try_from(transfer_fee_bps).unwrap());
        let key = Pubkey::new_unique();
        let mut lamports = u64::MAX;
        let token_program = spl_token_2022::id();
        let epoch = Clock::get().unwrap().epoch;
        let mint_info = AccountInfo::new(
            &key,
            false,
            false,
            &mut lamports,
            &mut mint_data,
            &token_program,
            false,
            Epoch::default(),
        );

        let amt = 100_u64;

        let owner_and_host_fee = fees.owner_trading_fee(amt.into()).unwrap();
        let host_fee = fees.host_fee(owner_and_host_fee).unwrap() as u64;
        let owner_fee = owner_and_host_fee.saturating_sub(host_fee as u128) as u64;
        let vault_amount = amt.saturating_sub(owner_and_host_fee as u64) as u64;

        let mint_data = mint_info.data.borrow();
        let mint =
            StateWithExtensions::<anchor_spl::token_2022::spl_token_2022::state::Mint>::unpack(
                &mint_data,
            )
            .unwrap();
        let transfer_fee_config = mint.get_extension::<TransferFeeConfig>().unwrap();
        let host_fee_xfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, to_u64!(host_fee).unwrap())
            .unwrap();
        let owner_fee_xfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, to_u64!(owner_fee).unwrap())
            .unwrap();
        let vault_amount_xfer_fee = transfer_fee_config
            .calculate_epoch_fee(epoch, to_u64!(vault_amount).unwrap())
            .unwrap();

        let host_feeee = host_fee.saturating_sub(host_fee_xfer_fee);
        let owner_feeee = owner_fee.saturating_sub(owner_fee_xfer_fee);
        let vault_feeee = vault_amount.saturating_sub(vault_amount_xfer_fee);

        let amount_re = amt
            .saturating_sub(host_fee_xfer_fee)
            .saturating_sub(owner_fee_xfer_fee)
            .saturating_sub(vault_amount_xfer_fee);
        let owner_and_host_fee_re = fees.owner_trading_fee(amount_re.into()).unwrap();
        let host_fee_re = fees.host_fee(owner_and_host_fee_re).unwrap() as u64;
        let owner_fee_re = owner_and_host_fee_re.saturating_sub(host_fee_re as u128) as u64;
        let vault_amount_re = amount_re.saturating_sub(owner_and_host_fee_re as u64) as u64;

        let host_fee_readd_xfer_fee = add_inverse_transfer_fee2(&mint_info, host_fee_re).unwrap();
        let owner_fee_readd_xfer_fee = add_inverse_transfer_fee2(&mint_info, owner_fee_re).unwrap();
        let vault_amt_readd_xfer_fee =
            add_inverse_transfer_fee2(&mint_info, vault_amount_re).unwrap();

        assert_eq!(host_fee, host_fee_readd_xfer_fee);
        assert_eq!(owner_fee, owner_fee_readd_xfer_fee);
        assert_eq!(vault_amount, vault_amt_readd_xfer_fee);

        let host_fee_sub_xfer_fee = host_fee.saturating_sub(host_fee_xfer_fee);

        let host_fee_readd_xfer_fee =
            add_inverse_transfer_fee2(&mint_info, host_fee_sub_xfer_fee).unwrap();

        assert!(
            host_fee_readd_xfer_fee <= host_fee,
            "host_fee_readd_xfer_fee: {}, host_fee: {}",
            host_fee_readd_xfer_fee,
            host_fee
        );
    }

    proptest! {
        #[test]
        fn test_sub_input_fees_always_favours_pool_by_at_most_two_or_three(
            amount in 1..u32::MAX as u64,
            owner_trade_fee_numerator in 0..100_000_u64,
            owner_trade_fee_denominator in 1..100_000_u64,
            host_fee_numerator in 0..100_000_u64,
            host_fee_denominator in 1..100_000_u64,
            transfer_fee_bps in 0..10_000_u64,
            host_fees: bool,
        ) {
            prop_assume!(host_fee_numerator <= host_fee_denominator);
            prop_assume!(owner_trade_fee_numerator <= owner_trade_fee_denominator);
            test_syscall_stubs();

            let mut mint_data = mint_with_fee_data();
            mint_with_transfer_fee(&mut mint_data, u16::try_from(transfer_fee_bps).unwrap());

            let key = Pubkey::new_unique();
            let mut lamports = u64::MAX;
            let token_program = spl_token_2022::id();
            let mint_info = AccountInfo::new(
                &key,
                false,
                false,
                &mut lamports,
                &mut mint_data,
                &token_program,
                false,
                Epoch::default(),
            );

            let fees = Fees {
                owner_trade_fee_numerator,
                owner_trade_fee_denominator,
                host_fee_numerator,
                host_fee_denominator,
                ..Default::default()
            };

            let amount_sub_fees = sub_input_transfer_fees(&mint_info, &fees, amount, host_fees).unwrap();
            // Compare with subtracting all fees at once
            let full_amount_sub_fees = sub_transfer_fee2(&mint_info, amount).unwrap();

            if host_fees {
                // At most a difference of 3 due to rounding from 3 transfers - 1 to the pool, 1 to the owner fees vault, 1 to the host account
                assert!(amount_sub_fees <= full_amount_sub_fees && full_amount_sub_fees - amount_sub_fees <= 3, "\nfull_amount_sub_fees should be greater than amount_sub_fees by at most 3.\namount={}\namount_sub_fees={}\nfull_amount_sub_fees={}\n", amount, amount_sub_fees, full_amount_sub_fees);
            } else {
                assert!(amount_sub_fees <= full_amount_sub_fees && full_amount_sub_fees - amount_sub_fees <= 2, "\nfull_amount_sub_fees should be greater than amount_sub_fees by at most 2.\namount={}\namount_sub_fees={}\nfull_amount_sub_fees={}\n", amount, amount_sub_fees, full_amount_sub_fees);
            }
        }
    }

    fn mint_with_transfer_fee(mint_data: &mut [u8], transfer_fee_bps: u16) {
        let mut mint =
            StateWithExtensionsMut::<spl_token_2022::state::Mint>::unpack_uninitialized(mint_data)
                .unwrap();
        let extension = mint.init_extension::<TransferFeeConfig>(true).unwrap();
        extension.transfer_fee_config_authority = OptionalNonZeroPubkey::default();
        extension.withdraw_withheld_authority = OptionalNonZeroPubkey::default();
        extension.withheld_amount = 0u64.into();

        let epoch = Clock::get().unwrap().epoch;
        let transfer_fee = TransferFee {
            epoch: epoch.into(),
            transfer_fee_basis_points: transfer_fee_bps.into(),
            maximum_fee: u64::MAX.into(),
        };
        extension.older_transfer_fee = transfer_fee;
        extension.newer_transfer_fee = transfer_fee;

        mint.base.decimals = 6;
        mint.base.is_initialized = true;
        mint.base.mint_authority = COption::Some(Pubkey::new_unique());
        mint.pack_base();
        mint.init_account_type().unwrap();
    }

    fn mint_with_fee_data() -> Vec<u8> {
        vec![
            0;
            ExtensionType::get_account_len::<spl_token_2022::state::Mint>(&[
                ExtensionType::TransferFeeConfig
            ])
        ]
    }
}
