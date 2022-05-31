To download the FGVC-Aircraft dataset, simply run:

```
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz

tar -xvzf fgvc-aircraft-2013b.tar.gz
```
(or a windows equivalent of these commands)

For the StanfordCars, access is granted through auth, so simply using `wget` won't work. You'll have to go to http://ai.stanford.edu.ezproxy2.utwente.nl/~jkrause/cars/car_dataset.html and download the tar of all images (and the bounding boxes and labels for both training and test data)
