![Static Badge](https://img.shields.io/badge/Python-3.8-blue) 
![coverage](https://img.shields.io/badge/Coverage-0%25-red)

# SQUAD v2.0 Tutorial

A sample collection of notebooks focused in solving the SQUAD v2.0 using traditional machine learning as well as embedding approaches.

# Demo

[SQUAD QA](http://ec2-16-170-163-223.eu-north-1.compute.amazonaws.com)

# Contents

* qa_squad A package that implements various operations related to Q models.
  * Implements a BiDAF Model.
  * Implements a dataset class for BiDAF.
  * Implements a BERT for QA training script.
  * Important environment variables
    * HF_TOKEN : Huggingface write key.
* model_api A simple API for future models.
  * Provides the endpoint squad_qa that returns an answer using a custom or a pretrained model.
  * Important environment variables
    * HF_TOKEN : Huggingface read key.
    * MODEL_ID : The model of the custom model.
    * DEVICE : The device to run the models. Default is 'cpu'.
* backend A simple backend that hosts the solution.
  * Provides a simple template using bootstrap.
  * Allows the user to run a simple query.

# Tests

Tests will be supported using PyTest in the future.

Coverage: 0%

# How to deploy?

First of all update the environment variables in dompose.yaml.

```bash
docker compose up --build
```

The app will be available on `http://localhost:80`.

# Under Construction

Some details will be added soon!