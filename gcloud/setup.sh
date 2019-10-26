gcloud pubsub topics delete section1_input
gcloud pubsub topics delete section1_input_delay
gcloud pubsub topics delete section1_backprop
gcloud pubsub topics delete section2_input
gcloud pubsub topics delete section2_input_delay
gcloud pubsub topics delete section2_backprop
gcloud pubsub topics delete section3_input
gcloud pubsub topics delete labels
gcloud pubsub topics create section1_input
gcloud pubsub topics create section1_input_delay
gcloud pubsub topics create section1_backprop
gcloud pubsub topics create section2_input
gcloud pubsub topics create section2_input_delay
gcloud pubsub topics create section2_backprop
gcloud pubsub topics create section3_input
gcloud pubsub topics create labels
mkdir section1
cp section1.py section1/main.py
cp requirements.txt section1
cd section1
gcloud functions deploy section1 --runtime python37 --entry-point run_section_forward --trigger-topic section1_input --set-env-vars BUCKET_NAME=dnn-module,MODEL_NAME=section1_model --memory=512MB
gcloud functions deploy section1_back --runtime python37 --entry-point run_section_backwards --trigger-topic section1_backprop --set-env-vars BUCKET_NAME=dnn-module,MODEL_NAME=section1_model --memory=512MB
cd ..
rm -rf section1
mkdir section2
cp section2.py section2/main.py
cp requirements.txt section2
cd section2
gcloud functions deploy section2 --runtime python37 --entry-point run_section_forward --trigger-topic section2_input --set-env-vars BUCKET_NAME=dnn-module,MODEL_NAME=section2_model --memory=512MB
gcloud functions deploy section2_back --runtime python37 --entry-point run_section_backwards --trigger-topic section2_backprop --set-env-vars BUCKET_NAME=dnn-module,MODEL_NAME=section2_model --memory=512MB
cd ..
rm -rf section2
mkdir section3
cp section3.py section3/main.py
cp requirements.txt section3
cd section3
gcloud functions deploy section3 --runtime python37 --entry-point run_section_full --trigger-topic section3_input --set-env-vars BUCKET_NAME=dnn-module,MODEL_NAME=section3_model --memory=1024MB
cd ..
rm -rf section3