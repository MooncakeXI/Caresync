use crate::Classification;
use prost::Message;
use std::cell::RefCell;
use tract_onnx::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

thread_local! {
    static MODEL: RefCell<Option<Model>> = RefCell::new(None);
}

/// An image classification model trained on ImageNet dataset.
/// The model itself is not included in the source code of this example.
/// If you see a compile error here, then download the model from:
/// https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet
/// See the `download_model.sh` script for details.
const IMAGENET: &'static [u8] = include_bytes!("../assets/model.onnx");

/// Constructs a runnable model from the serialized ONNX model in `IMAGENET`.
pub fn setup() -> TractResult<()> {
    let bytes = bytes::Bytes::from_static(IMAGENET);
    let proto: tract_onnx::pb::ModelProto = tract_onnx::pb::ModelProto::decode(bytes)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;
    MODEL.with_borrow_mut(|m| {
        *m = Some(model);
    });
    Ok(())
}

/// Runs the model on the given image and returns top three labels.
pub fn classify(image: Vec<u8>) -> Result<Vec<Classification>, anyhow::Error> {
    MODEL.with_borrow(|model| {
        let model = model.as_ref().unwrap();
        let image = image::load_from_memory(&image)?.to_rgb8();

        // The model accepts an image of size 224x224px.
        let image =
            image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);

        // Preprocess the input according to
        // https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet#preprocessing.
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        let tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
            (image[(x as u32, y as u32)][c] as f32 / 255.0 - MEAN[c]) / STD[c]
        });

        let result = model.run(tvec!(Tensor::from(tensor).into()))?;

        let mut scores: Vec<_> = result[0]
            .to_array_view::<f32>()?
            .into_iter()
            .zip(0..)
            .collect();

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let labels = scores
            .iter()
            .take(3)
            .map(|(score, i)| Classification {
                label: LABELS[*i as usize].to_string(),
                score: **score,
            })
            .collect();
        Ok(labels)
    })
}

/// The set of 1000 ImageNet class labels.
const LABELS: [&'static str; 4] = [
    "Hand Foot and Mouth Disease",
    "MonkeyPox",
    "Normal",
    "Tinea Ringworm Candidiasis",
   
];
