use support::{decl_storage, decl_module, StorageValue, StorageMap, dispatch::Result, ensure, decl_event};
use system::ensure_signed;
use runtime_primitives::traits::{As, Hash};
use parity_codec::{Encode, Decode};


#[derive(Encode, Decode, Default, Clone, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct NN_Model<Hash, Balance> {
    id: Hash,
    Weights: i64,
    Intercept: i64,
    LearningRate: i64,
    Loss: i64,
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
        ToFloat get(to_float) : i64 = 1_000_000_000;
        Prediction get(get_prediction): i64;
    }
}

decl_module! {
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {
        fn deposit_event<T>() = default;

        fn create_model(origin, weight: i64, intercept: i64, learnRate: i64) -> Result {
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
                Loss: 0 as i64,
                Bounty: <T::Balance as As<u64>>::sa(0),
            };

            <Nonce<T>>::mutate(|n| *n += 1);
            Self::commit_initial_model(sender, random_hash, new_model)?;

            Ok(())
        }
        fn update_model(origin, model_id: T::Hash, new_weight: i64, new_intercept: i64, new_learnRate: i64) -> Result {
            let sender = ensure_signed(origin)?;
            ensure!(<Models<T>>::exists(model_id), "This model doesnt exit");
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

        fn train_model(origin, model_id: T::Hash, data: i64, classification: i64) -> Result {
            let sender = ensure_signed(origin)?;
            ensure!(<Models<T>>::exists(model_id), "This model doesnt exit");
            //Get model
            let mut model = Self::stored_model(&model_id);
            let to_float = Self::to_float();
            let mut prediction = data * model.Weights + model.Intercept;
            let mut _update = (data * model.LearningRate / to_float) as i64;
            let mut new_weights: i64 = 0;
            let mut total_loss =  (classification - prediction).pow(2)/2 as i64;
            let mut _norm: i64 = 0;
            // Compute gradients
            let  d_total_loss_d_out = - (prediction - classification); 
            let  d_out_d_w = data;
            let  d_total_loss_d_w = d_total_loss_d_out * d_out_d_w; //using chain rule
            
            // Update weights
            new_weights = new_weights - model.LearningRate * d_total_loss_d_w;

            //Compute new loss
            let new_loss = (classification - (data * new_weights + model.Intercept)).pow(2)/2 as i64;

            // Commit new model

            model.Weights = new_weights;
            model.Loss = new_weights;
            <Models<T>>::insert(&model_id, model);
            
            // if classification > 0 {
            //     prediction = data * model.Weights + model.Intercept;
            //     new_weights = model.Weights + _update;
            //     _norm = _norm + data * data;
            // } else {
            //     // sign -1
            //     prediction = data * model.Weights + model.Intercept;
            //     new_weights = model.Weights - _update;
            //     _norm = _norm + data * data;
            // }

            // if prediction <= 0{
            //     prediction = 0;
            // } else {
            //     prediction = 1;
            // }
            
            // Must be almost within `toFloat` of `toFloat*toFloat` because we only care about the first `toFloat` digits.

            // let mut oneSquared = to_float * to_float;
            // let offset = to_float * 100;
            // ensure!(oneSquared - offset < _norm && _norm < oneSquared + offset, "The provided data does not have a norm of 1.");

            // if prediction != classification {
            //     model.Weights = new_weights;
            //     <Models<T>>::insert(&model_id, model);
            // }

            Ok(())
        }

        fn predict (origin, model_id: T::Hash, data: i64) -> Result {
            let sender = ensure_signed(origin)?;
            ensure!(<Models<T>>::exists(model_id), "This model doesnt exit");
            //Get model
            let mut model = Self::stored_model(&model_id);
            let m = model.Weights * data + model.Intercept;

            <Prediction<T>>::put(m);
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
}

decl_event!(
    pub enum Event<T>
    where
        <T as system::Trait>::AccountId,
        <T as system::Trait>::Hash,
        <T as balances::Trait>::Balance,
    {
        Created(AccountId, Hash),
        BountySet(AccountId, Hash, Balance),
        Predict(Hash, u64),
    }
);