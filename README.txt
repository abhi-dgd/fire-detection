I. Steps to recreate model training
    1. Setup environment using provided YML file
    2. Edit paths according to your system configuration
    3. Download labelled training data 
        from [https://www.kaggle.com/datasets/atulyakumar98/test-dataset]
        into cloned repository's data folder
    4. Check path setup once again. (Note: predict folder with data should not be
        in the same folder as training data.)
    5. Activate environment
    6. Start training using the training.py script

II. Steps to test model on random unseen images
    1. Download random images containing a fire or not in them
        from the internet
        into cloned repository's predict folder
    2. Standard size should be 1280 * 720
    3. Images will get resized when fed to predict
    4. Activate environment
    5. Check if .keras model is stored in model folder after training
    6. Check path setup once again. (Note: predict folder with data should not be
        in the same folder as training data.)
    7. Start testing using the testing.py
    8. Random images are automatically indexed from folder and provided to model
        to test
