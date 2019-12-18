use support::{decl_storage, decl_module, StorageValue, StorageMap, dispatch::Result, ensure, decl_event};
use system::ensure_signed;
use runtime_primitives::traits::{As, Hash};
use parity_codec::{Encode, Decode};


#[derive(Encode, Decode, Default, Clone, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct NN_Model<Hash, Balance> {
    id: Hash,
    Weights: u64,
    Intercept: u64,
    LearningRate: u64,
    Bounty: Balance,
}

pub trait Trait: balances::Trait {
    type Event: From<Event<Self>> + Into<<Self as system::Trait>::Event>;
}

decl_storage! {
    trait Store for Module<T: Trait> as Neurochain {
        
        Models get(stored_model): map T::Hash => NN_Model<T::Hash, T::Balance>;
        ModelOwner get(model_owner): map T::AccountId => T::Hash;

        // ParticipantsArray get(participant): map (T::AccountId, u64) => T::Hash;
        // ParticipantsCount get(num_of_participants): map T::AccountId => u64;
        // ParticipantsIndex: map T::Hash => u64;

        Nonce: u64;
    }
}

decl_module! {
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {
        fn deposit_event<T>() = default;

        fn create_model(origin, weight: u64, intercept: u64, learnRate: u64) -> Result {
            let sender = ensure_signed(origin)?;
            let nonce = <Nonce<T>>::get();
            let random_hash = (<system::Module<T>>::random_seed(), &sender, nonce)
                .using_encoded(<T as system::Trait>::Hashing::hash);

            ensure!(!<Models<T>>::exists(random_hash), "Model already exists!");

            let new_model = NN_Model {
                id: random_hash,
                Weights: weight,
                Intercept: intercept,
                LearningRate: learnRate,
                Bounty: <T::Balance as As<u64>>::sa(0),
            };

            <Nonce<T>>::mutate(|n| *n += 1);
            Self::commit_initial_model(sender, random_hash, new_model)?;

            Ok(())
        }
        fn update_model(origin, model_id: T::Hash, new_weight: u64, new_intercept: u64, new_learnRate: u64) -> Result {
            //Get model
            let mut model = Self::stored_model(&model_id);
            model.Weights = new_weight;
            model.Intercept = new_intercept;
            model.LearningRate = new_learnRate;
            //insert updated model
            <Models<T>>::insert(&model_id, model);

            Ok(())
        }
        fn set_bounty(origin, model_id: T::Hash, new_bounty: T::Balance) -> Result {
            let sender = ensure_signed(origin)?;
            ensure!(<Models<T>>::exists(model_id), "This model doesnt exit");

            let hash = Self::model_owner(&sender);
            ensure!(hash == model_id, "You didnt deploy this model");

            //Get model
            let mut model = Self::stored_model(&model_id);
            model.Bounty = new_bounty;

           //insert updated model
           <Models<T>>::insert(&model_id, model);
            Self::deposit_event(RawEvent::BountySet(sender, model_id, new_bounty));

            Ok(())
        }
    }
}
impl<T: Trait> Module<T> {
    fn commit_initial_model(creator: T::AccountId, model_id: T::Hash, model: NN_Model<T::Hash, T::Balance>) -> Result {

        // let number_of_participants = Self::num_of_participants();
        // let new_number_of_participants = number_of_participants.checked_add(1).ok_or("Overflow adding model")?;

        <Models<T>>::insert(model_id, model);
        <ModelOwner<T>>::insert(&creator, model_id);
        
        // <ParticipantsArray<T>>::insert((to.clone(), new_number_of_participants), model_id);
        // <ParticipantsCount<T>>::insert(&to, new_number_of_participants);
        // <ParticipantsIndex<T>>::insert(model_id, new_number_of_participants);
        
        Self::deposit_event(RawEvent::Created(creator, model_id));
        Ok(())
    }

    // fn predict_model() -> Result{
    //     Ok(())
    // }

    // fn update_model() ->Result{
    //     Ok(())
    // }
}

decl_event!(
    pub enum Event<T>
    where
        <T as system::Trait>::AccountId,
        <T as system::Trait>::Hash,
        <T as balances::Trait>::Balance
    {
        Created(AccountId, Hash),
        BountySet(AccountId, Hash, Balance),
    }
);