use solana_program::{program_option::COption, program_pack::Pack, pubkey::Pubkey};
use spl_token_2022::{
    extension::{BaseStateWithExtensions, ExtensionType, StateWithExtensions},
    state::{Account as TokenAccount, AccountState as TokenAccountState, Mint},
};

use crate::native_account_data::NativeAccountData;

pub fn create_mint(owner: &Pubkey, decimals: u8) -> NativeAccountData {
    let mut account_data = NativeAccountData::new(Mint::LEN, spl_token::id());
    let mint = Mint {
        is_initialized: true,
        mint_authority: COption::Some(*owner),
        decimals,
        ..Default::default()
    };
    Mint::pack(mint, &mut account_data.data[..]).unwrap();
    account_data
}

pub fn create_token_account(
    mint_account: &mut NativeAccountData,
    token_program: &Pubkey,
    owner: &Pubkey,
    amount: u64,
) -> NativeAccountData {
    create_token_account_with_address(
        mint_account,
        token_program,
        &Pubkey::new_unique(),
        owner,
        amount,
    )
}

pub fn create_token_account_with_address(
    mint_account: &mut NativeAccountData,
    token_program: &Pubkey,
    address: &Pubkey,
    owner: &Pubkey,
    amount: u64,
) -> NativeAccountData {
    let mut mint = Mint::unpack(&mint_account.data).unwrap();
    let mut account_data = NativeAccountData::new_with_key(
        *address,
        get_token_account_space(token_program, mint_account),
        *token_program,
    );
    let account = TokenAccount {
        state: TokenAccountState::Initialized,
        mint: mint_account.key,
        owner: *owner,
        amount,
        ..Default::default()
    };
    mint.supply += amount;
    Mint::pack(mint, &mut mint_account.data[..]).unwrap();
    TokenAccount::pack(account, &mut account_data.data[..]).unwrap();
    account_data
}

pub fn get_token_account_space(token_program: &Pubkey, mint: &NativeAccountData) -> usize {
    if token_program == &spl_token_2022::id() {
        // calculate the space for the token account with required extensions
        let mint = StateWithExtensions::<Mint>::unpack(&mint.data).unwrap();
        let mint_extensions: Vec<ExtensionType> =
            BaseStateWithExtensions::get_extension_types(&mint).unwrap();

        let required_extensions =
            ExtensionType::get_required_init_account_extensions(&mint_extensions);

        ExtensionType::get_account_len::<TokenAccount>(&required_extensions)
    } else {
        TokenAccount::LEN
    }
}

pub fn get_token_balance(account_data: &NativeAccountData) -> u64 {
    let account = TokenAccount::unpack(&account_data.data).unwrap();
    account.amount
}

pub fn transfer(
    from_account: &mut NativeAccountData,
    to_account: &mut NativeAccountData,
    amount: u64,
) {
    let mut from = TokenAccount::unpack(&from_account.data).unwrap();
    let mut to = TokenAccount::unpack(&to_account.data).unwrap();
    assert_eq!(from.mint, to.mint);
    from.amount -= amount;
    to.amount += amount;
    TokenAccount::pack(from, &mut from_account.data[..]).unwrap();
    TokenAccount::pack(to, &mut to_account.data[..]).unwrap();
}
