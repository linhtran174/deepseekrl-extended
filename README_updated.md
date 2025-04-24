## Dependencies
- `pip install -r requirements.txt`
- Install latest transformer package from source (to include Qwen2_5_VLForConditionalGeneration)
pip install git+https://github.com/huggingface/transformers accelerate

## Run command
python main.py     \
--model_name Qwen/Qwen-VL-Chat \
--dataset_name flowers \
--output_dir ./qwen_vl_flowers_output \
--learning_rate 1e-6 \
--kl_weight_beta 0.1 \
--num_train_iters 3000 \
--eval_iterations 50 \
--temperature 0.9