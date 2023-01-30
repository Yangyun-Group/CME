train_file=./data/cme_final_soho
python3 code/SVM_train.py  --input_file ${train_file}_train.json --model_save_path ./model/SVM_model
python3 -u code/XGBoost_train.py --input_file ${train_file}.json --train_epochs 1000 --model_save_path ./model/XGB_model
