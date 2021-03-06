
- [ ] come up with distractor loss functions 
- [x] define interesting distractor internal architecture
- [x] write pseudocode
- [x] pick initial dataset
- [ ] implement pseudocode
- [ ] define assessment method for heatmaps (aside from distractor loss function)
- [ ] describe dataset
  - [ ] total sample count
  - [ ] classes, class distribution
  - [ ] colour distribution...?

TODO Implementations:
- [ ] Preperations
  - [ ] Import datasets (Stanford Cars and FGVC-Aircraft)
  - [ ] Finetuning models (ResNet-50 and VGG16)
  - [ ] Generating and saving attention maps

- [ ] Distractor (in PyTorch)
  - [ ] Architecture
  - [ ] Training

- [ ] Retraining FT-models
  - [ ] 
  - [ ] 


## Distractor loss functions
- standard deviation of the resulting heatmap
- variance of the resulting heatmap
- ...? think of more

## define distractor internal architecture(s)
### input parameters
- original image (optional)
- Grad-CAM heatmap


### output options
1. occlusion mask, to overlay onto original image
2.  directly occluded image
  - may require an intermediate loss? idk, might be quite complex

### layer structure options
1. dense 3 layer feedforward
2. Encoder decoder (conv)
- requires thought to be put into latent dimensions, and depth of encoder and decoder + decoder dimensions
3. ?

### overall structure/pipeline

<!-- ![structure_drawing.svg](../../REDI_architecture.drawio.svg) -->


## pseudocode
```python
# basic training loop
for batch in dataset:
    outputs = classifier(batch)
    loss = L(outputs, batch_ground_truths)
    loss.backprop()

# heatmap collection & distractor training loop
while True:
    while not distractor.converged:
        for batch in dataset: # entire dataset? or only misclassified samples?
            outputs = classifier(batch)
            heatmaps = heatmaps(classifier, batch)
            occlusion_masks = distractor(batch, heatmaps)
            occluded_batch = occlude(batch, occlusion_masks)
            new_heatmaps = heatmaps(classifier, occluded_batch)

            distractor_loss = L_distractor(new_heatmaps, etc...)
            distractor_loss.backprop()

    while not classifier.converged:
        for batch in dataset: # entire dataset? or only misclassified samples?
            outputs = classifier(batch)
            heatmaps = heatmaps(classifier, batch)
            occlusion_masks = distractor(batch, heatmaps)
            occluded_batch = occlude(batch, occlusion_masks)
            new_outputs = classifier(occluded_batch)
            
            classifier_loss = L_classifier(new_outputs, occluded_batch_ground_truths)
            classifier_loss.backprop()
```