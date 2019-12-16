use support::{decl_storage, decl_module, StorageValue, StorageMap, dispatch::Result, ensure, decl_event};
use system::ensure_signed;
use runtime_primitives::traits::{As, Hash};
use parity_codec::{Encode, Decode};

#[derive(Encode, Decode, Default, Clone, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct NN_Model<Hash, AccountId> {
    id: Hash,
    Creator: AccountId,
    Weights: map u64=>u64,
    Intercept: map u64=>u64,
    LearningRate: u64;
}

pub trait Trait: balances::Trait {
    type Event: From<Event<Self>> + Into<<Self as system::Trait>::Event>;
}

decl_storage! {
    trait Store for Module<T: Trait> as Neurochain {
       
        ToFloat: 1e9;

        Models get(model): map T::Hash => NN_Model<T::Hash, T::Balance);
        ModelCreator get(model_owner): map T::Hash => Option<T::AccountId>;

        ParticipantsArray get(participant): map (T::AccountId, u64) => T::Hash;
        ParticipantsCount get(num_of_participants): map T::AccountId => u64;
        ParticipantsIndex: map T::Hash => u64;

        Nonce: u64;
    }
}

decl_module! {
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {

        fn deposit_event<T>() = default;

        fn create_model(origin, weights: vec![], intercept: vec![], learnRate: u64) -> Result {
            let sender = ensure_signed(origin)?;

            let nonce = <Nonce<T>>::get();
            let random_hash = (<system::Module<T>>::random_seed(), &sender, nonce)
                .using_encoded(<T as system::Trait>::Hashing::hash);

            ensure!(!<Models<T>>::exists(random_hash), "Model already exists!");

            let new_model = NN_Model {
                id: random_hash,
                Weights: weights,
                Intercept: intercept,
                LearningRate: learnRate,
            };

            <Nonce<T>>::mutate(|n| *n += 1);
            Self::mint(sender, random_hash, new_kitty)?;
            Ok(())
        }

        fn update_weights(origin)-> Result {

            Ok(())
        }

    }
}
impl<T: Trait> Module<T> {
    fn commit_initial_model(to: T::AccountId, model_id: T::Hash, model: NN_Model<T::Hash, T::AccountId>) -> Result {

        let number_of_participants = Self::num_of_participants();
        let new_number_of_participants = number_of_participants.checked_add(1).ok_or("Overflow adding model")?;

        <Model<T>>::insert(model_id, model);
        <ModelCreator<T>>::insert(model_id, &to);
        
        <ParticipantsArray<T>>::insert((to.clone(), new_number_of_participants), model_id);
        <ParticipantsCount<T>>::insert(&to, new_number_of_participants);
        <ParticipantsIndex<T>>::insert(model_id, new_number_of_participants);
        
        Self::deposit_event(RawEvent::Created(to, model_id));

        Ok(())
    }
}

decl_event!(
    pub enum Event<T>
    where
        <T as system::Trait>::AccountId,
        <T as system::Trait>::Hash
    {
        Created(AccountId, Hash),
    }
);