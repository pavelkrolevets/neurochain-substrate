pub struct NN_Model_regres {
    Weights: Vec<u8>,
    Intercept: i64,
    LearningRate: i64,
    Loss: i64,
}

fn train_model_regression(model: &NN_Model_regres, data: &Vec<u8>, Y: i64) -> Result<NN_Model, ParseIntError> {

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
        w = w - ((prediction - Y) / model.LearningRate * data[i] as i64) as i64;
        updated_weights.push(w as u8);
    }

    // Compute gradients  and update intercept
    let mut updated_intercept: i64 = model.Intercept;
    updated_intercept -= ((prediction - Y) / model.LearningRate) as i64 ;

    //Compute new error
    let mut new_prediction: i64 = updated_intercept;
    for i in 0..updated_weights.len() {
        new_prediction +=  data[i] as i64 * updated_weights[i] as i64;
    }
    let new_loss = (new_prediction - Y).pow(2)/2 as i64;

    // Commit new model
    let updated_model = NN_Model{
        Weights: updated_weights,
        Loss: new_loss,
        Intercept: updated_intercept,
        LearningRate: model.LearningRate
    };

    Ok(updated_model)
}

fn predict (model: &NN_Model_regres, data: &Vec<u8>) -> Result<i64, ParseIntError> {

    let mut m = model.Intercept ;
    for i in 0..model.Weights.len(){
        m = m + model.Weights[i] as i64 * data[i] as i64;
    }

    Ok(m)
}
