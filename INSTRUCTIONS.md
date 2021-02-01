![AIAP Banner](../imgs/AIAP-Banner.png "AIAP Banner")

<center><h1>Week 7: Deployment</h1></center>

<center>It is recommended that you spend <b>5</b> days for this assignment</center>

## 0. Introduction

Welcome to Week 7 of the coursework! In this week, we will look at deploying one of the models we have built in the previous weeks - the TensorFood model 😋. In some projects, you may be required to build an end-to-end product including creating an app to allow your users the ability to consume your model. This is sometimes known as full stack development.

The goal for this week is to serve the model through a web application at `http://<your-username>.aiap.okdapp.tekong.aisingapore.net` such that you can upload an image of some food and obtain its prediction. This involves understanding web serving, web-based apps, HTTP, containerisation and CI/CD. We'll also make you write test cases, implement logging and write documentation. There are other things that we want you to pick up along the way but won't be mentioned explicitly (like testing if things work locally first).

To get started, first run `git checkout master` and then `git pull origin master`. Then create a new branch `deploy-<your-name>` and check out that branch `git checkout deploy-<your-name>`. The entire assignment should be completed on this new branch. The CI/CD build pipeline specific to this week only gets triggered with branches that start with `deploy-`.

**Important: Read the `CODE_OF_CONDUCT.md` which outlines our expectations for all apprentices at AI Singapore for this assignment.**

#### Objectives

- Able to deploy an AI/ML app with Flask and Docker
- Understand HTTP basics
- Able to make HTTP calls using `curl`, Postman and a Python client
- Understand importance of unit testing and logging
- Understand the basics of CI/CD pipeline
- Write good documentation

#### Topics

- Web server
- HTTP
- Web-based applications & HTML
- Unit testing
- Logging
- Containerisation
- CI/CD
- Documentation

#### Deliverables

1. The following files, pushed to your GitLab branch (**DO NOT commit your model artefacts**):

    ```text
    assignment7
    ├── env.sh
    ├── README.md
    ├── Dockerfile
    ├── diagram.png
    ├── conda.yml
    ├── CODE_OF_CONDUCT.md
    ├── INSTRUCTIONS.md    (this)
    ├── skaffold.yaml      (don't edit)
    ├── ci                 (don't edit)
    ├── src
    │   ├── __init__.py
    │   ├── app.py
    │   └── inference.py
    └── tests
        ├── __init__.py
        └── test_inference.py
    ```

2. Your app at `http://<your-username>.aiap.okdapp.tekong.aisingapore.net/` where `your-username` is your GitLab username without underscore or dash. No action is needed for this part (read on to find out why).

    Do remember to credit yourself on your page, as it may be retained as an example for future batches 😊.
    Here's an example done by an apprentice from a previous batch:
    http://demo.aiap.okdapp.tekong.aisingapore.net/.

## 1. Retrieve trained model

The first step to build your TensorFood app is to retrieve the trained model. Obtain the download URL of the TensorFood model.

1. In your browser, open the experiment with the saved model on Polyaxon.
2. Click **Outputs** tab.
3. Select `tensorfood.h5`. A download button should appear on the right (An icon with a cloud and a downward arrow. The icon **does not** have the word 'Download')
4. Get the URL to the saved model by using a right click on the button and copying the URL.

In the `env.sh` file, enter this URL in the `MODEL_URL` variable and the filename in the `MODEL_NAME` variable.

### On GitLab runner

Since the model will not be uploaded to the repository, the CI/CD pipeline has been [configured](../.gitlab-ci.yml) to download the model onto the GitLab runner at the beginning of each pipeline run. The model will then be available in the following stages of the pipeline.

The model will be downloaded from `MODEL_URL` and saved as `MODEL_NAME` in the `assignment7` directory. **Your code should access the model from this path.**

### On your local machine

You will need to download a copy of the model to your machine for local testing. You may use the `curl` command below (replacing `model-url` and `your-token` with your model URL and Polyaxon token respectively):

```bash
curl --request GET '<model-url>' --header 'Authorization: token <your-token>'
```

<blockquote>
<details>
<summary>Follow these steps to get your Polyaxon token:</summary>

1. Log in to your account at http://polyaxon.okdapp.tekong.aisingapore.net/.
2. On the top navigation bar, click the icon with the initial of your name.
3. In the dropdown, click **Personal token**. You will be taken to a page that displays your token, which is a base64 string.

</details>
</blockquote>

Store your local model to the **same path** that the model will be [downloaded](#on-gitlab-runner) to in the CI/CD pipeline, so that your local testing reflects the way the model will be accessed in the pipeline.

Note: You should **NOT** include your token (or any other secret keys) in any of your commits. Your personal token should only be used for the download to your local machine. The GitLab runner will use its own token to pull the model from Polyaxon. (For the curious, the token is stored as a [CI variable](https://docs.gitlab.com/ee/ci/variables/) on GitLab - you can observe its use in [deploy.sh](../deploy.sh)).

## 2. Prepare an inference script

The next step is to write your inference module or script, named `src/inference.py`. This module contains inference functions and utilities related to tensor manipulation or image size checks - we leave these to you. However, you should design your module such that

1. calling `python -m src.inference your_test_image.png` from the interpreter will give you a prediction, and
2. its functions can be used by other modules, i.e. adopt a modular design (you'll see why later).

(Hint: use the `if __name__ == '__main__'` idiom.)

### Test locally

Before going to the next section, you are to test if this module works locally by running `python -m src.inference your_test_image.png`.

## 3. Write unit tests

Writing test cases is an important part of software development, like in [TDD](https://en.wikipedia.org/wiki/Test-driven_development). For this assignment, we will use the `pytest` framework. You will write 3 test cases in `tests/test_inference.py` that will test the functions that you have created in `src/inference.py`. These tests will be run in the CI/CD pipeline.

### Test locally

Before going to the next section, run `pytest` within the `assignment7` directory to see if your tests pass.

## 4. Create a Flask app

In order for others to use your model which resides in your computer, other machines or devices (computers, mobile phones etc.) need to be able to communicate with it using a communication model or protocol. Enter HTTP.

### HTTP & API

Read up about HTTP and [API][1] and make sure you can answer these questions:

- What is the client-server model?
- What is HTTP?
- What are GET and POST HTTP methods?
- What does the body and header of an HTTP GET and POST request look like?
- What is meant by an API?

### Flask app

To build an API, you will leverage on the popular Python framework [`Flask`](3). Flask is Python's abstraction for communicating with the web server, allowing you to quickly create web-based applications. Here's an example Flask app serving a model from the PyTorch tutorials: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html.

Your task is to write a Flask app in `src/app.py`. In your app, you are expected to:

- Provide 3 API endpoints:

    1. **Model information (frontend)**

        ```text
        GET /
        ```

        **Response**

        A simple HTML document which contains information about the model and training. Ensure the content is decent but you can get playful with the design - we leave it up to your creativity ✌🏻.

    2. **Model information**

        ```text
        GET /info
        ```

        **Response**

        Returns information about the model and what input it expects in JSON. Below is an example (add more things you deem pertinent).

        ```json
        {
            "model": "MobileNetv3",
            "input-size": "224x224x3",
            "num-classes": 13,
            "pretrained-on": "ImageNet"
        }
        ```

    3. **Model inference**

        ```text
        POST /predict
        ```

        **Request**

        Body of HTML should include the image as the payload

        **Response**

        Returns the prediction and its probability in JSON.

        ```json
        {
            "food": "tau_huay",
            "probability": 0.92
        }
        ```

- Load the model accordingly.

- Leverage the functions and utilities from the `src.inference` module to perform the inference.

- In production mode, make sure to use the production-level web server and serve your app on port `8000` at `0.0.0.0`.

### Example

These should also demonstrate what an HTTP client is and the difference between GET and POST methods.

http://demo.aiap.okdapp.tekong.aisingapore.net/

http://demo.aiap.okdapp.tekong.aisingapore.net/info

http://demo.aiap.okdapp.tekong.aisingapore.net/predict

```bash
curl --request GET 'http://demo.aiap.okdapp.tekong.aisingapore.net/'
```

```bash
curl --request GET 'http://demo.aiap.okdapp.tekong.aisingapore.net/info'
```

```bash
curl --location \
    --request POST 'http://demo.aiap.okdapp.tekong.aisingapore.net/predict' \
    --form 'file=@<path_to_image>'
```

### Test locally

Before going to the next section, you are to test if this app works locally by running `python -m src.app` and using tools like [`curl`][9] and [Postman][4]. You may also get your peers to access your endpoint using the local network.

## 5. Logging

If you have been using `print()` to log your app, it's time to switch to the builtin `logging` module. Decide for yourselves what should be logged while the app is running. Below is an example log statement that informs us when the model is loaded for the first time while running our app:

```text
[2020-01-01 05:51:27,703] INFO in app: Loading model for the first time
```

## 6. Dockerise the Flask app

We will now containerise your app using Docker. Here is a quick guide:

1. Install Docker for your computer.
2. Prepare a `conda.yml` which defines the required dependencies of your app.
3. Update the `Dockerfile` that we have provided such that:
    - The base image `aiap/polyaxon/pytorch-tf2-cpu` with tag `latest` is pulled from the registry `registry.aisingapore.net`.
    - The required files that your app needs are copied into the image. **This includes your TensorFood model with filename `MODEL_NAME` (that's right, we want the model to be built in the image!).**
    - The conda environment in the image will be updated based on the `conda.yml`.
    - The Flask app starts once the container is run.

### Test locally

Before going to the next section, you are to ensure that the Docker image can be built and the container with the app can run locally. 

**Important: You are not required to and are not authorized to push any image directly to `registry.aisingapore.net`. This will automatically be done within the CI/CD pipeline.**

## 7. CI/CD

As you already know, every time you push your commits to your branch, a **CI/CD pipeline** is triggered, which is dictated by `.gitlab-ci.yml`. And for this assignment, you won't pass the CI/CD pipeline unless you have some files and performed some tests 🤭.

For this section:

1. Read up on CI/CD. GitLab has a good starting point [here](https://about.gitlab.com/ci-cd/). Understand why we use it.
2. Read up our `.gitlab-ci.yml`. Do check out what we test for in the CI/CD pipeline. Once you're good to go, you can write all about it in the next section 😉.

(Note: `skaffold` and `Kubernetes` are not part of the curriculum so you may like to skim through it.)

## 8. Documentation

Documentation is extremely important to ensure that when we pass our code on to another analyst or engineer, they will know how to work with and modify the code for their own use.

For this assignment, you are expected to write a `README.md`. Look through ML projects on GitHub, have a sense of how READMEs are usually structured, and write one yourself. Some pointers you should include (but not limited to):

- How the model is trained
- The expected format the model requires
- Details about the dataset with which the model was trained
- Performance of the model
- How the model is served

On top of that, to demonstrate that you have understood the CI/CD pipeline, we require that you include a diagram `diagram.png` in the `README.md` that shows the flow of CI/CD pipeline and indicate the following:

- What is considered integration?
- What is considered deployment?

You may also like to host your README at `/docs`! Here's an example: http://demo.aiap.okdapp.tekong.aisingapore.net/docs.

## 9. Build and host your app

At this point of time, it is highly likely that you already have all the required files for this week's deliverables. Perform a `git push` (**DO NOT commit your model**) to your deployment branch, which will trigger the CI/CD pipeline which involves building the image in the GitLab runner server, pushing it to our internal container registry and triggering a series of steps that builds your app in the Tekong cluster.

If all is well, you should be able to see your app hosted at `http://<your-username>.aiap.okdapp.tekong.aisingapore.net/` 🎉🥳. Congrats!

(The exact application URL can be found in the CI/CD pipeline logs if the build was successful)

## 10. Building a Python client

Test your app and your peers'!

Previously, you tested the REST API using either the command line (`curl`) and/or an external tool (Postman). Creating a client can help you understand how your API (and therefore your model) will be used by your users. This in turn affects how you design your API. Furthermore, building your client in Python will allow you to test your API endpoints in a consistent manner.

And now to the task at hand: build a simple client in Python, `python_client.py`. To complete this task, you are required to do the following:

- Load in an image.
- Using libraries like `http`, [`requests`](https://requests.readthedocs.io/en/master/) or [`urlib3`](https://urllib3.readthedocs.io/en/latest/), call the appropriate APIs (your peers' and yours) to classify the image.
- Display the image and overlay the result to the user, using any Python imaging library that you are familiar with.

## 11. Improving your app

There are more things that you can do to improve your app! Here are some suggestions:

- Is your model getting loaded at every HTTP POST request?
- How do you make your model run inference faster?
- Create a nicer welcome page.
- Create a simple user interface that lets users upload images and obtain the result, like this example: http://demo.aiap.okdapp.tekong.aisingapore.net/.
- Host `README.md` together in the app by serving it on another route, say `/docs`.
- How would you design the CI/CD pipeline?

[1]:https://www.mulesoft.com/resources/api/what-is-an-api/
[2]:https://www.mulesoft.com/resources/api/what-is-rest-api-design
[4]:https://www.getpostman.com/
[5]:https://www.docker.com/resources/what-container
[6]:https://docker-curriculum.com/
[7]:https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest
[8]:http://gitlab.int.aisingapore.org/aisg/base-docker-images
[9]:https://www.booleanworld.com/curl-command-tutorial-examples/
