use support::{decl_storage, decl_module, StorageValue, StorageMap, dispatch::Result, ensure, decl_event, StorageList};
use system::ensure_signed;
use runtime_primitives::traits::{As, Hash};
use runtime_primitives::{Perbill, Permill};
use parity_codec::{Encode, Decode};
use rstd::prelude::*;

#[derive(Encode, Decode, Default, Clone, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct NN_Model<Hash, Balance> {
    id: Hash,
    Weights: Vec<u8>,
    Intercept: i64,
    LearningRate: i64,
    Loss: i64,
    Bounty: Balance,
}

pub trait Trait: balances::Trait {
    type Event: From<Event<Self>> + Into<<Self as system::Trait>::Event>;
}

pub fn matrix_dot(a: Vec<Vec<i64>>, b: Vec<Vec<i64>>)-> Vec<Vec<i64>>{
    // get sizes
    let (a_row_len, a_col_len, b_row_len, b_col_len) = (a.len(), a[0].len(), b.len(), b[0].len());
    // check if its possible to do dot multiplication
    if a_col_len != b_row_len {
        panic!("Matricies are not compatible");
    }

    let mut result: Vec<Vec<i64>> = vec!(vec!(0; b_col_len); a_row_len); 
    for i in 0..a_row_len {
        for j in 0..b_col_len{
            let mut sum: i64 = 0;
            for k in 0..a_col_len {
                sum += a[i][k] * b[k][j];
            // println!("i,j,k {}{}{}", i,j,k);
            }
        result[i][j] = sum;
        }
    }
    result   
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
        Test: Vec<Vec<u8>>;
    }
}

decl_module! {
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {
        fn deposit_event<T>() = default;

        fn create_model(origin, weights: Vec<u8>, intercept: i64, learnRate: i64) -> Result {
            let sender = ensure_signed(origin)?;
            let nonce = <Nonce<T>>::get();
            let random_hash = (<system::Module<T>>::random_seed(), &sender, nonce)
                .using_encoded(<T as system::Trait>::Hashing::hash);

            // let weights_slice: &[u8] = &[1u8, 1u8, 1u8];
            // let weight: Vec<u8> = weights_slice.iter().cloned().collect();
            
            ensure!(!<Models<T>>::exists(random_hash), "Model already exists!");

            let new_model = NN_Model {
                id: random_hash,
                Weights: weights,
                Intercept: intercept,
                LearningRate: learnRate,
                Loss: 0 as i64,
                Bounty: <T::Balance as As<u64>>::sa(0),
            };

            <Nonce<T>>::mutate(|n| *n += 1);
            Self::commit_initial_model(sender, random_hash, new_model)?;

            Ok(())
        }

        fn update_model(origin, model_id: T::Hash, new_weights: Vec<u8>, new_intercept: i64, new_learnRate: i64) -> Result {
            let sender = ensure_signed(origin)?;
            ensure!(<Models<T>>::exists(model_id), "This model doesnt exit");
            //Get model
            let mut model = Self::stored_model(&model_id);
            model.Weights = new_weights;
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

        fn train_model(origin, model_id: T::Hash, data: Vec<u8>, classification: i64) -> Result {
            let sender = ensure_signed(origin)?;
            ensure!(<Models<T>>::exists(model_id), "This model doesnt exit");
            //Get model
            let mut model = Self::stored_model(&model_id);
            let to_float = Self::to_float();
            // Check weights lengh == data.lengh
            ensure!(model.Weights.len() == data.len(), "Data provided dont have same dimentions with weights.");

            let mut prediction = model.Intercept;
            let mut new_weights: Vec<u8> = Vec::new();
            let mut _norm: i64 = 0;
            
            if classification > 0 {
                for i in 0..model.Weights.len() {
                    let dataum = data[i] as i64;
                    let w = model.Weights[i] as i64;
                    prediction = prediction + dataum * w;
                    new_weights.push((w + (dataum * model.LearningRate / to_float) as i64) as u8);
                    _norm = _norm + (dataum * dataum);
                }
                
            } else {
                // sign -1
                for i in 0..model.Weights.len() {
                    let dataum = data[i] as i64;
                    let w = model.Weights[i] as i64;
                    prediction = prediction + dataum * w;
                    new_weights.push((w - (dataum * model.LearningRate / to_float) as i64) as u8);
                    _norm = _norm + (dataum * dataum);
                }
            }

            if prediction <= 0{
                prediction = 0;
            } else {
                prediction = 1;
            }
            
            //Must be almost within `toFloat` of `toFloat*toFloat` because we only care about the first `toFloat` digits.

            let mut oneSquared = to_float * to_float;
            let offset = to_float * 100;
            ensure!(oneSquared - offset < _norm && _norm < oneSquared + offset, "The provided data does not have a norm of 1.");

            if prediction != classification {
                model.Weights = new_weights;
                <Models<T>>::insert(&model_id, model);
            }
            Ok(())
        }

        fn predict (origin, model_id: T::Hash, data: Vec<u8>) -> Result {
            let sender = ensure_signed(origin)?;
            ensure!(<Models<T>>::exists(model_id), "This model doesnt exit");
        
            //Get model
            let mut model = Self::stored_model(&model_id);

            // Check weights lengh == data.lengh
            ensure!(model.Weights.len() == data.len(), "Data provided dont have same dimentions with weights.");

            let mut m = model.Intercept ;
            for i in 0..model.Weights.len(){
                m = m + model.Weights[i] as i64 * data[i] as i64;
            }

            <Prediction<T>>::put(m);

            Ok(())
        }
        
        fn multiply_float(origin) -> Result {
           
           
            <Test<T>>::put(c);
            Ok(())
        }

        fn train_model_regression(origin, model_id: T::Hash, data: Vec<u8>, Y: i64) -> Result {
            let sender = ensure_signed(origin)?;
            ensure!(<Models<T>>::exists(model_id), "This model doesnt exit");

            //Get model
            let mut model = Self::stored_model(&model_id);
            // let to_float = Self::to_float();
            let mut error: i64 = 0;
            
            let mut prediction: i64 = model.Intercept;
            // Compute error
            for i in 0..model.Weights.len(){
                prediction += data[i] as i64 * model.Weights[i] as i64;
            }
            error = (prediction - Y).pow(2)/2 as i64;

            // Compute gradients and update weights
            let mut updated_weights: Vec<u8> = Vec::new();

            let mut w: i64 = 0;
            for i in 0..model.Weights.len(){
                w = w - (prediction - Y) * model.LearningRate * data[i] as i64;
                updated_weights.push(w as u8);
            }
            
            // Compute gradients  and update intercept
            let mut updated_intercept: i64 = model.Intercept;
            updated_intercept -= (prediction - Y) * model.LearningRate;

            //Compute new error
            let mut new_prediction: i64 = updated_intercept;
            for i in 0..updated_weights.len() {
                new_prediction +=  data[i] as i64 * updated_weights[i] as i64;
            }
            let new_loss = (new_prediction - Y).pow(2)/2 as i64;

            // Commit new model

            model.Weights = updated_weights;
            model.Loss = new_loss;
            model.Intercept = updated_intercept;
            <Models<T>>::insert(&model_id, model);
            Ok(())
        }
    }
}
impl<T: Trait> Module<T> {
    fn commit_initial_model(creator: T::AccountId, model_id: T::Hash, model: NN_Model<T::Hash, T::Balance>) -> Result {

        // let number_of_participants = Self::num_of_participants();
        // let new_number_of_participants = number_of_participants.checked_add(1).ok_or("Overflow adding model")?;

        <Models<T>>::insert(&model_id, model);
        <ModelOwner<T>>::insert(&creator, &model_id);
        
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