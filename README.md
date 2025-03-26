I implemented and evaluated the image models VGG16, ResNet (20/56/164), AlexNet, and DenseNet using metamorphic testing techniques. Specifically, I applied four methods: Reject Differential, Majority Voting, Plurality Voting, and Weighted Voting. These techniques assess how stable model predictions remain under minor input perturbations, using image transformations like rotation, brightness changes, and translation. The goal is to determine the reliability and consistency of a model's predictions when faced with slight input changes.

Reject Differential: Compares confidence of predictions between the original and transformed inputs.  The output is the absolute difference in softmax confidence scores.
Goal: Detect sensitivity of model predictions to slight input perturbations.

Majority Voting: Accepts prediction if a majority of transformed versions agree with the original prediction. The output is accuracy among samples where original = majority vote.

Plurality Voting: Accepts prediction if it matches the most frequent prediction from the transformed set. The output is accuracy and acceptance rate where prediction equals the plurality vote.

Weighted Voting: Assigns custom weights ω to each transformed model prediction.  The prediction is accepted only if it matches the weighted vote outcome. The output is the accuracy and acceptance rate among accepted samples.

How to Run:

# Initialize and train
model = ResNet20().to(device)
for epoch in range(num_epochs):
    fit(model, trainloader)
    validate(model, testloader)

Run RejDiff:

# Outputs batch-wise reject differential
run_reject_differential(model, testloader, transformations)

Run MajVot:

test_with_majority_vote(model, testloader, transformations, device)

Run PluVot:

test_with_plurality_vote(model, testloader, transformations, device)

Run WeightVot:

weights = [0.5, 0.25, 0.25, 0.25]  # Must sum to 1
test_with_weighted_vote(model, testloader, transformations, weights, device)

Transformations used:

These are small perturbations that should not affect class identity:

RandomRotation(degrees=10)

ColorJitter(brightness=0.2)

RandomAffine(degrees=0, translate=(0.1, 0.1))





