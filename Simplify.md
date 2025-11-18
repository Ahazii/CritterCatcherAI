OK we can dramatically simplify this.

The Purpose of this app is to try and record a small group of different animals. For example Hedgehogs.

The user should select a few YOLO categories that the animals could possibly fall under, and type in the name of the animal they want to record. As there is no category for Hedgehog, the user will need to choose multiple categories of animals that could look like their requested animal (in this example, Hedgehogs are small animals with four legs and spikes, so could look like cat, dog, sheep maybe cow.)

When processing the videos, the system should pass any videos that are identified as possibly in the selected groups, onto another AI image analyser that can more accurately identify the requested animal (in our example, Hedgehogs).

If the second stage AI image analysis decides that it is definitely a hedgehog then the video should be stored into the folder /data/sorted. Any videos that it is not sure about should be saved for the operator to confirm as correctly identified with "Yes" or "no" - This confirmation should feed back into the AI system to train it further to identify the animals that the user wants (in this example Hedgehogs)

The User should be able to select more than one type of animal - in my example they want to record hedgehogs and birds.

Or would it be better to just use the one AI system that is able to identify the Hedgehogs to start with?