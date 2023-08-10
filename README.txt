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
    1. Activate environment
    2. Check if .keras model is stored in model folder after training
    3. Check path setup once again. (Note: predict folder with data should not be
        in the same folder as training data.)
    4. Start testing using the testing.py
    5. Random images from the predict folder are automatically indexed
        and provided to model for inference
