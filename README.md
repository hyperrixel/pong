# pong


**Pong** is an AI and ML based artwork recommendation system

The [demo](https://github.com/hyperrixel/pong#demo) recommends **27.898** artworks 🖼 based on **2074** [keywords](https://github.com/hyperrixel/pong#one-hot-encoded-keywords) 💬, **7** different colorfulness categories 🌈 or outputs of AI 🧠 or ML 🗜 models. 

---


## Inspiration 💡

Recommendation is part of our everyday life. We can meet AI, ML or simple algorithm based recommendation systems all the time from choosing music or movies through the ideal program and washing time on washing machines till the autocorrect service of different messaging apps.The museum or artwork recommendation seems to be still in the developing phase. 🚀

We want to develop a system that is easily maintainable and the cost of operation is low because there is no need to use cloud infrastructure or services. More details under the [business](https://github.com/hyperrixel/pong#business) section.


## Concept 🧑‍🎨

**Pong** provides artwork-level prediction based on different approaches. Since **pong** already contains 2 different AI models and an ML method, the exact approach depends on which subsystem is use:
- ✔ [pong vision AI](https://github.com/hyperrixel/pong#pong-vision-ai)
- ✔ [pong content AI](https://github.com/hyperrixel/pong#pong-content-ai)
- 🚧 [pong complex AI](https://github.com/hyperrixel/pong#pong-complex-ai) [under development]
- ✔ [pong vector ML](https://github.com/hyperrixel/pong#pong_vector_ml)

The data structures are very similar. We created one-hot encoded matrices from the data. The core of the matrices are the same for each AI or ML model, but each model adds extra records to its own data-matrix.

****Example to show one-hot encoding method:**** 
Note: This example is to demonstrate how see a one-hot encoded matrix, the dataset of pong contains different fields

|  |   |   |   |   |   |   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| -- | animal | fruit | green | dangerous | extinct | imaginary |
| 🐶 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| 🦝 | 🟩 | 🟥 | 🟥 | 🟩 | 🟥 | 🟥 |
| 🐍 | 🟩 | 🟥 | 🟩 | 🟩 | 🟥 | 🟥 |
| 🦖 | 🟩 | 🟥 | 🟥 | 🟩 | 🟩 | 🟥 |
| 🦕 | 🟩 | 🟥 | 🟥 | 🟥 | 🟩 | 🟥 |
| 🦄 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 | 🟩 |
| 🐲 | 🟩 | 🟥 | 🟥 | 🟩 | 🟥 | 🟩 |
| 🍍 | 🟥 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |
| 🍓 | 🟥 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |
| 🍏 | 🟥 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |
| 🍄 | 🟥 | 🟩 | 🟥 | 🟩 | 🟥 | 🟥 |

## Data 💾

The test dataset comes from the [Badische Landesmuseum](https://www.landesmuseum.de/). Due to the legal limitation of the dataset, the link to the demo page is private only and the dataset is not uploaded to this github repository as well. If you want to try out the code, please make a contact with the museum for getting access to the data.

### Cleaning the data 🧹

The original dataset contains images and additional information about artworks. The additional information was in json format. We decided to filter the data based on multiple conditions:
1. since this is a recommendation system, each records is dropped without image
2. the following features will be selected: ` source_id `, ` sammlung `, ` objektid `, ` objekttitel_de `, `objektbezeichnung `, `material `,  `technik `, ` medium `, `schlagworte`, `beschreibung_de`
3. `schlagworte`, `beschreibung_de` are not necessary fields, but all other fields are mandatory
4. one-hot encoded records made from `material `,  `technik `, ` medium ` features
5. if the frequency of keyword is higher than 15, that keyword is putted into the one-hot encoded list
6. create colorfulness value based on the Measuring Colourfulness in Natural Images [paper](https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf) by Hasler-Süsstrunk

### Shape of dataset 📐

After the cleaning process the dataset contains **27.898** artworks 🖼 sliced by **2074** base tags 💬 and 7 different colorfulness categories 🌈.

---


## How it works ⚙

The process contains two parts:
1. [warm-up phase](https://github.com/hyperrixel/pong#warm-up-phase)
2. [recommendation phase](https://github.com/hyperrixel/pong#recommendation-phase)


### warm-up phase 🦾

This phase is for the model to get to know the user’s preferences. During this phase we show **10** random artworks and the user has to rate all of them with one of our [rating methods](https://github.com/hyperrixel/pong#rating-methods). We provide two different scales, each of them containing 5 different categories to rate. The rating categories are:
- [reaction based rating](https://github.com/hyperrixel/pong#reaction-based-rating): 🙁 😕 😐 🙂 😍
- [personal impact based rating](https://github.com/hyperrixel/pong#personal-impact-based-rating): ⭐ ⭐ ⭐ ⭐ ⭐

Each category is represented by a unique value point to help the evaluation. 


### recommendation phase 📊

In this phase the system uses the knowledge from warm-up to recommend artworks. The user has the chance to rate artworks, and based on that rating indirectly the recommendation algorithm as well, to have better recommendations. Our system watches these feedbacks. It is important that each artwork is evaluated even if the user does not click on any emojis, since the next navigation arrow ➡ is represented by a unique value. This method provides the variability of recommendations. When an image makes an impact on the user, they will rate it with one of the emojis or stars. Therefore when the user just clicks on the next arrow, the system knows the image makes no impact to the user. After some no impact images the recommendation system will evaluate artworks based on other features. 
The recommendation algorithms are below:
- [pong vision AI](https://github.com/hyperrixel/pong#recommendation-algorithm-of-pong-vision-ai)
- [pong content AI](https://github.com/hyperrixel/pong#recommendation-algorithm-of-pong-content-ai)
- [pong vector ML](https://github.com/hyperrixel/pong#recommendation-algorithm-of-pong-vector-ml)

---


## AI & ML 🤖


### pong vision AI

The goal of **pong** vision AI is to provide recommendations based on the visual content. Each artwork is represented by a vector that contains ` 2048 ` values. This vector is the output of the bottleneck fully connected layer of an AutoEncoder neural network. The network consists from 3 parts:
1. Encoder layers
2. Bottleneck layer
3. Decoder layers

The autoencoding method is widely used in different fields where an AI has to recognize things or distinguish them, such as person identification by face recognition, fingerprint identification or tumor detection on screening images (like MRI).

The network learns only from visual data, which means, but not limited to shapes, colors, contrast, brightness. So basically all visual information that can get from an image with convolutional layers and kernels.

There is under development an AI which is the cross of **pong** vision AI and **pong** content AI. It is called [pong complex AI](https://github.com/hyperrixel/pong#pong-complex-ai) [under development].


#### Architecture

The core concept of the encoder’s architecture comes from the widely used pretrained [VGG-networks](https://arxiv.org/abs/1409.1556v6)’ architectures. The images about artwork were resized with fixed ratio as the shortest side of the image should be equal with 224 pixels. The next phase was a center crop to get the ` 224 x 224 ` image size. All images are normalized between 0 and 1 to converge faster.

The network:

**Encoder layers**
```
CNN1 : Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
 BN1 : BatchNorm2d(64)
  MP : MaxPool2d(kernel_size=2, stride=2)
 ACT : LeakyReLU
CNN2 : Conv2d(64, 128, 3, stride=1, padding=1, bias=True)
 BN2 : BatchNorm2d(128)
  MP : MaxPool2d(kernel_size=2, stride=2)
 ACT : LeakyReLU
CNN3 : Conv2d(128, 256, 3, stride=1, padding=1, bias=True)
 BN3 : BatchNorm2d(256)
  MP : MaxPool2d(kernel_size=2, stride=2)
 ACT : LeakyReLU
CNN4 : Conv2d(256, 512, 3, stride=1, padding=1, bias=True)
 BN4 : BatchNorm2d(512)
  MP : MaxPool2d(kernel_size=2, stride=2)
 ACT : LeakyReLU
CNN5 : Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
 BN5 : BatchNorm2d(512)
  MP : MaxPool2d(kernel_size=2, stride=2)
 ACT : LeakyReLU
CNN6 : Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
 BN6 : BatchNorm2d(512)
  MP : MaxPool2d(kernel_size=2, stride=2)
 ACT : LeakyReLU
CNN7 : Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
 BN7 : BatchNorm2d(512)
  MP : MaxPool2d(kernel_size=2, stride=2)
 ACT : LeakyReLU
```

**Bottleneck layer**
```
 FC1 : Linear(512, 2048, bias=True)
 ACT : TanH
```

**Decoder layers**
```
TCNN1 : ConvTranspose2d(512, 512, 3, stride=2, padding=1, bias=True)
 DBN1 : BatchNorm2d(512)
   UP : Interpolate(scale_factor=1.5, recompute_scale_factor=True)
  ACT : LeakyReLU
TCNN2 : ConvTranspose2d(512, 512, 3, stride=2, padding=0, bias=True)
 DBN2 : BatchNorm2d(512)
   UP : Interpolate(scale_factor=2.5, recompute_scale_factor=True)
  ACT : LeakyReLU
TCNN3 : ConvTranspose2d(512, 512, 3, stride=2, padding=1, bias=True)
 DBN3 : BatchNorm2d(512)
   UP : Interpolate(scale_factor=2)
  ACT : LeakyReLU
TCNN4 : ConvTranspose2d(512, 512, 3, stride=1, padding=0, bias=True)
 DBN4 : BatchNorm2d(512)
   UP : Interpolate(scale_factor=2)
  ACT : LeakyReLU
TCNN5 : ConvTranspose2d(512, 256, 3, stride=1, padding=1, bias=True)
 DBN5 : BatchNorm2d(256)
   UP : Interpolate(scale_factor=2)
  ACT : LeakyReLU
TCNN6 : ConvTranspose2d(256, 128, 3, stride=1, padding=1, bias=True)
 DBN6 : BatchNorm2d(128)
   UP : Interpolate(scale_factor=2)
  ACT : LeakyReLU
TCNN7 : ConvTranspose2d(128, 64, 3, stride=1, padding=1, bias=True)
 DBN7 : BatchNorm2d(64)
   UP : Interpolate(scale_factor=2)
  ACT : LeakyReLU
TCNN8 : ConvTranspose2d(64, 3, 3, stride=1, padding=1, bias=True)
 DBN8 : BatchNorm2d(3)
   UP : Interpolate(scale_factor=2)
  ACT : LeakyReLU
TCNN8 : ConvTranspose2d(3, 3, 3, stride=1, padding=1, bias=True)
 DBN8 : BatchNorm2d(3)
  ACT : Sigmoid
```

Loss: **M**ean **A**bsolute **E**rror
Optimizer: ADAM with 1e-4 initial learning rate and own learning rate scheduler

After the training phase, the decoder layers can be dropped, since their purpose is only to train the bottleneck layer well. This method provides full flexibility as when a new artwork comes to the recommendation system, the image should be forwarded through the encoder and bottleneck layers and the output of the bottleneck layer should be saved and stored.

Here are some example images from the training process. The best results are not shown, since the original artworks can be recognized and this is over the legal limitation of the original dataset.

![Good example 1](https://github.com/hyperrixel/pong/blob/main/asset/good_example_1.png "Good example 1")
![Good example 2](https://github.com/hyperrixel/pong/blob/main/asset/good_example_2.png "Good example 2")
![Good example 3](https://github.com/hyperrixel/pong/blob/main/asset/good_example_3.png "Good example 3")
![Good example 4](https://github.com/hyperrixel/pong/blob/main/asset/good_example_4.png "Good example 4")

We show our bad experiments below, since we think a really good solution cannot be realized without experimenting. In this scope there are no failures, just a pure knowledge of how our goals cannot be achieved. Each test leads us closer to our goals. 

![Bad example 1](https://github.com/hyperrixel/pong/blob/main/asset/bad_example_1.png "Bad example 1")
![Bad example 2](https://github.com/hyperrixel/pong/blob/main/asset/bad_example_2.png "Bad example 2")
![Bad example 3](https://github.com/hyperrixel/pong/blob/main/asset/bad_example_3.png "Bad example 3")


#### Recommendation algorithm of pong vision AI

The base of recommendation is the output of the bottleneck layer, which is a vector that contains ` 2048 ` values between -1 and 1 due to the tanh activation function on the layer. The recommendation algorithm watches the last rating and calculates the mean absolute difference between each image and the lastly rated image. When the user gives a positive feedback, the recommendation algorithm returns the image with the lowest mean absolute difference. When the user gives a negative feedback, the recommendation algorithm returns the image with the highest mean absolute difference. When the user gives neutral feedback, the recommendation algorithm chooses randomly.

The recommendation algorithm is written in Javascript. This provides platform independent operation. However, the algorithm is easy to adopt into other languages like Java, Python or C++. 


### pong content AI

The goal of **pong** content AI is to provide recommendations based on the one-hot encoded features. Each artwork is represented by a vector that contains ` 384 ` values. This vector is the output of the bottleneck fully connected layer of an AutoEncoder neural network. The network consists from 3 parts:
1. Encoder layers
2. Bottleneck layer
3. Decoder layers

The concept is similar with  **pong** vision AI, but in this case the focus is on the description of artworks. From the view of description, there is no place for the detailed visual contents. 


#### Architecture

The input data is the 2074 [keywords](https://github.com/hyperrixel/pong#one-hot-encoded-keywords)  and the 7 different colorfulness categories. Since The data matrix contains zeros and ones, there is no need for any other preprocessing.

The network:

**Encoder layers**
```
FC1 : Linear(2081, 1024, bias=True)
BN1 : BatchNorm1d(1024)
ACT : LeakyReLU
FC2 : Linear(1024, 512, bias=True)
BN2 : BatchNorm1d(512)
ACT : LeakyReLU
DRP : Dropout(p=0.2)
FC3 : Linear(512, 512, bias=True)
BN3 : BatchNorm1d(512)
ACT : LeakyReLU
DRP : Dropout(p=0.2)
FC4 : Linear(512, 384, bias=True)
BN4 : BatchNorm1d(384)
ACT : LeakyReLU
DRP : Dropout(p=0.2)
```

**Bottleneck layer**
```
FCB : Linear(384, 384, bias=True)
ACT : TanH
```

**Decoder layers**
```
FC5 : Linear(384, 512, bias=True)
BN5 : BatchNorm1d(512)
ACT : LeakyReLU
DRP : Dropout(p=0.2)
FC6 : Linear(512, 512, bias=True)
BN6 : BatchNorm1d(512)
ACT : LeakyReLU
DRP : Dropout(p=0.2)
FC7 : Linear(512, 1024, bias=True)
BN7 : BatchNorm1d(1024)
ACT : LeakyReLU
DRP : Dropout(p=0.2)
FC8 : Linear(1024, 2081, bias=True)
ACT : Sigmoid
```

Loss: **M**ean **A**bsolute **E**rror
Optimizer: ADAM with 1e-4 initial learning rate and own learning rate scheduler


#### Recommendation algorithm of pong content AI

The base of recommendation is the output of the bottleneck layer, which is a vector that contains ` 384 ` values between -1 and 1 due to the tanh activation function on the layer. The recommendation algorithm watches the last rating and calculates the mean absolute difference between each image and the lastly rated image. When the user gives a positive feedback, the recommendation algorithm returns the image with the lowest mean absolute difference. When the user gives a negative feedback, the recommendation algorithm returns the image with the highest mean absolute difference. When the user gives neutral feedback, the recommendation algorithm chooses randomly.

The recommendation algorithm is written in Javascript. This provides platform independent operation. However, the algorithm is easy to adopt into other languages like Java, Python or C++. 


### pong complex AI

🚧 Under development

**Pong** complex AI is the combine of **pong** vision AI and **pong** content AI. Where the target vector contains visual and content data at the same time.


### pong vector ML

This is a simple feature selection method, where each user ratings are summed up together. The recommendation system always watches the ratings and calculates the most valued categories. Most valued means the highest average score from the user. This is a simple yet powerful solution.


## Rating methods

We implement two different rating methods. However, in the future we would like to make experiments with other methods. We are really curious about a recommendation based on real emotions or a recommendation where the rating is based on verbal feedback only. This last one means the integration of an ASR and an NLP module. 

Each rating system consists of 5 different rating categories. Each of them is represented by a unique value. These values are used by the recommendation algorithms to calculate the next recommended artwork.

The meaning of emojis do not matter. The purpose of them is only to give a mask to the values that they represent. When we created a basic machine learning model at the beginning of development, we used the following values: ` -3 `, ` -2 `, ` -1 `, ` 1 `, ` 2 `. The ` 0 ˙ value was reserved to the skipped images. Now we only use **6** values from the inclusive **0 to 5** scale. The difference between the two methods is only to have the chance for the user to choose a preferred one scale.


### Reaction based rating

Reaction categories :🙁 😕 😐 🙂 😍


### Personal impact based rating

Reaction categories :⭐ ⭐ ⭐ ⭐ ⭐


## One-hot encoded keywords

` Keramik `, ` getöpfert `, ` Märtyrer `, ` Essgeschirr `, ` Religion `, ` Silber `, ` getrieben `, ` gegossen `, ` ziseliert `, ` gehämmert `, ` Christusdarstellung `, ` Heiligenverehrung `, ` Jesus Christus `, ` Maria `, ` gelötet `, ` montiert `, ` geschmiedet `, ` graviert `, ` Gold `, ` gepunzt `, ` Trachtbestandteil `, ` Schmuck `, ` Kleidung `, ` Bronze `, ` Blei `, ` Messing `, ` Zinn `, ` Email <Beschichtung> `, ` Glas `, ` emailliert `, ` Getränk `, ` Kanne `, ` Gefäß `, ` Gewandschmuck `, ` Buntsandstein `, ` Steinmetzarbeit `, ` Bauwerk `, ` Tempel `, ` Göttin `, ` Gesellschaft `, ` geschlagen (Kerngerät) `, ` Einzelfund `, ` Werkzeug `, ` Schneidwerkzeug `, ` geritzt `, ` geschnitzt `, ` Tierdarstellung `, ` Siedlungsfund `, ` gegossen (Schalenguss) `, ` Hortfund `, ` Grabfund `, ` Flussfund `, ` gebrannt `, ` bemalt `, ` Brandgrab `, ` Urnengrab `, ` Hügelgrab `, ` gebogen `, ` Kopfschmuck `, ` poliert `, ` geschliffen `, ` Kelten `, ` Defensivwaffe `, ` Rüstung <Schutzkleidung> `, ` Grabbeigabe `, ` Zierstück `, ` Gürtel `, ` Sandstein `, ` Alltag `, ` Mobilität `, ` Staat `, ` Messgerät `, ` engobiert `, ` Eisen `, ` Metallverarbeitung `, ` Truhe `, ` Möbel `, ` Büste `, ` Wohnen `, ` Henkel `, ` Mond `, ` Pfau `, ` Bestattung `, ` Ernährung `, ` geblasen `, ` Landwirtschaftliches Gerät `, ` Landwirtschaft `, ` Gabel `, ` Sammlung Altertumsverein Sinsheim `, ` Perlmutter `, ` Lebensmittel `, ` Handwerk `, ` Holzwirtschaft `, ` Holzbearbeitung `, ` Holzhandwerk `, ` Militär `, ` Offensivwaffe `, ` Waffe `, ` Lanze `, ` Gerät `, ` Marmor `, ` gepresst `, ` Reiter `, ` Pferd `, ` Reitzeug `, ` Kult/Religion `, ` Zaumzeug `, ` geformt `, ` Halsschmuck `, ` Vergoldetes Silber `, ` nielliert `, ` Kerbschnitt `, ` Almandin `, ` vergoldet `, ` Einlagen `, ` Goldblech `, ` genietet `, ` Filigran `, ` Verschluss <Kleidung> `, ` geprägt `, ` Zahlungsmittel `, ` Geld `, ` Armschmuck `, ` gedreht `, ` Ess- und Trinksitte `, ` Deckel `, ` tauschiert `, ` verzinnt `, ` Schmieden `, ` Haarschmuck `, ` Blattgold `, ` Granat `, ` Schild `, ` Lebensmittelverarbeitung `, ` Bernstein `, ` Handel `, ` Silberblech `, ` Knochen (tierisch) `, ` gesägt `, ` gebohrt `, ` Hygiene `, ` Körperpflege `, ` Gürtelgarnitur `, ` Gravierung `, ` Vergoldung `, ` Kreuz `, ` Muscheln `, ` Hockergrab `, ` Felsgestein `, ` geschnitten `, ` gegossen (Kernguss) `, ` Anhänger <Schmuck> `, ` gewickelt `, ` Vorratshaltung `, ` Feuerstein `, ` geschlagen (Abschlaggerät) `, ` Amphibolit `, ` Lappenbeil `, ` Schwert `, ` Bogen <Waffe> `, ` Pfeil `, ` durchbohrt `, ` Textilhandwerk `, ` Transport `, ` Vögel `, ` Kochgeschirr `, ` Buchenholz `, ` Holz `, ` Victoria, Göttin `, ` Kaiser `, ` Überfangglas `, ` Fritte `, ` Fayence `, ` Steinzeug `, ` Weißmetall `, ` Bastfaser `, ` Punzarbeit `, ` Spiel `, ` versilbert `, ` Perle `, ` Kerbtechnik `, ` Bergkristall `, ` Kupfer `, ` Muschel `, ` Sammlung Pfähler `, ` Jagd `, ` Kochen `, ` Spinnen <Handwerk> `, ` Metallguss `, ` Elfenbein `, ` glasiert `, ` Löffel `, ` Beleuchtung `, ` geflochten `, ` Bildnis `, ` Ahornholz `, ` Brettspiel `, ` Korallen / Schmuckstein `, ` Sport `, ` Wirtschaft `, ` Verwahrung `, ` Silex `, ` Vogel `, ` Material fraglich `, ` Kunst `, ` Flachs `, ` Feuer `, ` Münze `, ` Klebstoff `, ` Ägypten `, ` Hahn `, ` Tierfigur `, ` Denar `, ` geklebt `, ` Stater `, ` geätzt `, ` Mythologie `, ` Medizin `, ` Kosmetik `, ` Beschlag `, ` Inschrift `, ` Schreibzeug `, ` Ackerbau `, ` Obst `, ` Aureus `, ` Herrscherbildnis `, ` Follis `, ` Malerei `, ` Freizeitgestaltung `, ` Unterhaltung `, ` Löwe `, ` Karneol `, ` gedrechselt `, ` Webstuhl `, ` Hieb- und Stoßwaffe `, ` Schuh `, ` Lapislazuli `, ` Gusseisen `, ` Design <1920er Jahre> `, ` Design <Deutschland> `, ` Staatliches Bauhaus `, ` Leuchte `, ` Metall `, ` Kunststoff `, ` Gummi `, ` Industriedesign `, ` Design <1950er Jahre> `, ` Küchengerät `, ` Elektrogerät `, ` Fernsehempfänger `, ` Stahl `, ` Bakelite `, ` Nationalsozialismus `, ` Hakenkreuz <Politisches Symbol> `, ` Pappe `, ` Fahrrad `, ` Design <1980er Jahre> `, ` Lackierung `, ` Filz <Textilien> `, ` Bemalung `, ` Wiener Werkstätte `, ` Sperrholz `, ` gebaut `, ` Leder `, ` Wolle `, ` gestanzt `, ` genäht `, ` Design <1960er Jahre> `, ` Accessoire `, ` Mode `, ` Leinen `, ` Papier `, ` Karton `, ` Art Déco `, ` Islam `, ` Moschee `, ` Migration `, ` Haushaltsartikel `, ` Spritzgießen `, ` Design <1970er Jahre> `, ` Design <2000er Jahre> `, ` Ring <Schmuck> `, ` Design <1930er Jahre> `, ` Design <Schweden> `, ` Design <Skandinavien> `, ` Kommunikation `, ` Horn `, ` Jugendstil `, ` Blech `, ` Spritzdekor `, ` Aluminium `, ` Badezimmer `, ` Design <1990er Jahre> `, ` Gips `, ` Maske `, ` Ritzdekor `, ` Glasblasen `, ` Gießen <Urformen> `, ` Glasschliff `, ` Sägen `, ` ABS-Kunststoff `, ` Design <Italien> `, ` Reiseutensil `, ` gedruckt `, ` Architektur `, ` Kristallglas `, ` Glasverarbeitung `, ` Silikon `, ` Elektronik `, ` Beton `, ` Edelstahl `, ` Kaffeekultur `, ` Brennen `, ` Leuchter `, ` Geschirr <Hausrat> `, ` Tischkultur `, ` Werbung `, ` Goldauflage `, ` Siebdruck `, ` Gewehr `, ` Emotional Design `, ` Textil `, ` vernickelt `, ` lackiert `, ` Stahlrohr `, ` Stahlblech `, ` Baumwolle `, ` Unikat `, ` Beleuchtungskörper `, ` Design <Karlsruhe> `, ` Ebenholz `, ` Polycarbonat `, ` Polypropylen `, ` Plexiglas `, ` Elefanten `, ` Wachs `, ` Reinigung / Utensil `, ` Staubsauger `, ` Nussbaumholz `, ` Profil `, ` Kunststoffguss `, ` Polyethylen `, ` Zink `, ` Hund `, ` Küchenarbeit `, ` Katze `, ` Kunststofffolie `, ` Stanzen `, ` Kreuzigung Jesu `, ` Kunstleder `, ` Kunstfaser `, ` Samt `, ` Nonne `, ` Baumwollfaser `, ` Fußball `, ` Trinkkultur `, ` Baumwollgewebe `, ` Reinigungsmittel `, ` Frühstück `, ` Lack `, ` Raumausstattung `, ` Karlsruhe / Staatliche Hochschule für Gestaltung / Edition kkaarrlls `, ` Preis <Auszeichnung> `, ` Orientalismus `, ` Tafelgerät `, ` Camping `, ` Multifunktion `, ` Kunsthandwerk `, ` Holzkunst `, ` Farbfassung `, ` Weichholz `, ` Österreich `, ` Marketerie `, ` Intarsie `, ` Staatspreis Baden-Württemberg für das Kunsthandwerk `, ` Besteck `, ` Kuchen `, ` Porzellan `, ` Acrylglas `, ` Weißblech `, ` Nylon `, ` Reise `, ` Photoapparat `, ` Phototechnik `, ` Küche `, ` Styrol-Acrylnitril-Copolymere `, ` Design <Finnland> `, ` Angewandte Kunst seit 1945 `, ` Schulkiste `, ` Spritzguss `, ` Design <Dänemark> `, ` Tür `, ` Polystyrol `, ` Blumen `, ` Kiefernholz `, ` Peddigrohr `, ` Flechten `, ` Recyclingprodukt `, ` Design <2010er Jahre> `, ` Upcyclingprodukt `, ` Pigment `, ` Kunsthandwerk <2010er Jahre> `, ` Birnbaumholz `, ` Hausarbeit `, ` Schwan `, ` Rundfunkempfänger `, ` Durmelit `, ` Deutsche Lufthansa `, ` Flugreise `, ` Melamin `, ` Tiere `, ` Schale <Gefäß> `, ` Karlsruhe / Staatliche Hochschule für Gestaltung `, ` Holzspielzeug `, ` Lehrmittel `, ` Teakholz `, ` Weide `, ` Flechtarbeit `, ` Stroh `, ` Bast `, ` Behältnismöbel `, ` Baumwollgarn `, ` Rauchutensilie `, ` Weidenrute `, ` Isolierkanne `, ` Plastik <Chemie> `, ` Blume `, ` Messingblech `, ` Rose `, ` Tunesien `, ` Fenster `, ` Mahagoni `, ` Kirche `, ` Bischof `, ` Reh `, ` Rehkitz `, ` Heimat `, ` Kaffee `, ` Extrudieren `, ` Pressen `, ` Uhr `, ` Halbedelstein `, ` Acryl `, ` Türkis `, ` Frömmigkeit `, ` Krippe `, ` Perlon `, ` Draht `, ` Baumstamm `, ` Baum `, ` Seide `, ` Strohflechterei `, ` Kupferblech `, ` Schneiden `, ` Fell `, ` Eichenholz `, ` Fräsen `, ` Kabbalistik `, ` Zeitgenössische Kunst `, ` Abendmahl `, ` Apostel `, ` Chalzedon `, ` Verpackung `, ` Hut `, ` gewebt `, ` Religiöse Druckgraphik `, ` gedrückt `, ` Elektrik `, ` Ton <Geologie> `, ` gemodelt `, ` Ritus `, ` Wallfahrt `, ` Gebetsstein `, ` Imam Hossein `, ` Schiiten `, ` Iran `, ` Nadelholz `, ` Schreinerarbeit `, ` Photopapier `, ` Nadelbaum `, ` Schellack `, ` Frau `, ` Corona-Pandemie `, ` Strohhut `, ` Mann `, ` Boot `, ` Party `, ` Temperafarbe `, ` Mann / Bildnis `, ` Frau / Bildnis `, ` Miniatur `, ` Schnitzerei `, ` Souvenir `, ` Exotismus `, ` Quarz-Frittenware `, ` gesprüngelt `, ` Inglasurmalerei `, ` Aufglasurmalerei `, ` Arabeske `, ` Rankenwerk `, ` Seldschuken `, ` Islamische Keramik `, ` Scharffeuerfarbe `, ` Irdengut / Scherben / Rot / Gelb `, ` Glasur `, ` Unterglasurmalerei `, ` Engobe `, ` Quattre-fleurs-Stil `, ` Tulpe `, ` Türkei `, ` Nelke `, ` Zinnglasur `, ` Japan `, ` Irdengut / Scherben / Rot `, ` China `, ` Drache `, ` Bleiglasur `, ` Ente `, ` Landschaft `, ` Buddhismus `, ` Teezeremonie `, ` Craquelé <Glasur> `, ` Teegeschirr `, ` Blüte `, ` Schlickermalerei `, ` Sonne `, ` Quarzkeramik `, ` Seladonglasur `, ` Seladon `, ` Schriftkunst `, ` Alkaliglasur `, ` Quarzfritte `, ` Kleinplastik `, ` Götter `, ` Emailmalerei `, ` Marke `, ` Glückssymbol `, ` Wappen `, ` Dekoration `, ` Osmanisches Reich `, ` Syrien `, ` Fliese `, ` Lüster `, ` Laufglasur `, ` Reliefdekor `, ` Chrysantheme `, ` Safawiden `, ` Chinoiserie `, ` Pagode `, ` Irdengut `, ` Töpferware `, ` Schmelzfarbe `, ` Famille rose `, ` Figurengruppe `, ` Flasche `, ` Vase `, ` Landschaftsbild `, ` Heiliger `, ` Liebespaar `, ` Genreszene `, ` Schmetterlinge `, ` Irdenware `, ` Figur `, ` Craquelé `, ` Temmoku `, ` Schrift `, ` Tee `, ` Blatt `, ` Terrakotta `, ` Kranich `, ` Ochsenblutglasur `, ` Goldbemalung `, ` Fels `, ` Kampf `, ` Fassmalerei `, ` Pfingstrose `, ` Schablone `, ` Fisch `, ` Marokko `, ` Qadjaren `, ` Krieger `, ` Interieur `, ` Familie `, ` Unterglasur `, ` Osmanen `, ` Blütenmotive `, ` Okzidentalismus `, ` Mariendarstellung `, ` Blumenarrangement `, ` Keramik `, ` Salzglasur `, ` Lüsterglasur `, ` Karlsruhe / Badische Sammlungen für Altertums- und Völkerkunde `, ` Bier `, ` Antiatlas `, ` Berber `, ` Alltagskultur `, ` Kind `, ` Feija `, ` Polyester `, ` gestickt `, ` Marokko (Süd) `, ` Farbmittel `, ` Alltagsgegenstand `, ` Animismus `, ` Apotropäon `, ` Baukeramik `, ` Wandfliese `, ` Schmiedekunst `, ` Schwertstichblatt `, ` Lackarbeit `, ` Sommer `, ` See `, ` Brücke `, ` Gebirge `, ` Teich `, ` Kirschblüte `, ` Fluss `, ` Wasser `, ` Fischer `, ` Wanderer `, ` Pflanzen `, ` Haus `, ` Silberdraht `, ` Steingut `, ` Palast `, ` Silberlahn `, ` Brokat `, ` Textilien `, ` Lampas `, ` Seidenweberei `, ` Seidenbrokat `, ` Hose `, ` Weintraube `, ` Weben `, ` Venedig `, ` Spanien `, ` Tughra `, ` Seidenseele `, ` Weltausstellung `, ` Balkanhalbinsel `, ` Seidensamt `, ` Italien `, ` Drucktechnik `, ` Kolonialismus `, ` Metallfaden `, ` Kostüm `, ` Theater `, ` bedruckt `, ` bestickt `, ` Münzen `, ` gefärbt `, ` Stoff <Textilien> `, ` Laubbaumholz `, ` Obi `, ` patiniert `, ` Seerose <Pflanze, Motiv> `, ` Fuchs `, ` Kastanie `, ` Ochse `, ` Affen `, ` Stein `, ` Achat `, ` Siegelstempel `, ` Siegel `, ` Kakishibu `, ` Kôzo `, ` Itoire `, ` Chûgata `, ` Tsukibori `, ` Katagami `, ` Japonismus `, ` Bambus `, ` Vierjahreszeiten `, ` Herbst `, ` Hütte `, ` Schiff `, ` Leimen `, ` Schriftsteller `, ` Papiermaché `, ` Tablett `, ` Bäuerin `, ` Gewässer `, ` Terrasse `, ` Berg `, ` Segelschiff `, ` Gewebe <Textilien> `, ` Pailletten `, ` Globalisierung `, ` Tasche `, ` Spiegel `, ` Frau / Wohnraum `, ` Literatur `, ` Pfeilköcher / Schutzhülle `, ` Treibarbeit `, ` Tanz `, ` Holzverarbeitung `, ` Musikinstrument `, ` Schnur `, ` Geldmittel `, ` Glasperle `, ` Paradies `, ` Hirsche `, ` Meer `, ` Hochzeit `, ` Schloss `, ` Burg `, ` Ufer `, ` Karaffe `, ` Theater / Aufführung `, ` Fest `, ` Requisit `, ` Schauspieler `, ` Bekleidungsbestandteil `, ` Haar `, ` Verkleidung `, ` Jacke `, ` Aufbaukeramik `, ` Feldbrand `, ` Berber / Keramik `, ` Berberin <Frau> `, ` Frauen aus Tunesien `, ` Frau / Menschenrasse `, ` Holzschnitt `, ` Gruppe `, ` Zeichnen `, ` Zeichnung `, ` Schnee `, ` Bauer `, ` Wasserfall `, ` Frühling `, ` Dorf `, ` Winter `, ` Kutsche `, ` Friedrich <Baden, Großherzog, I.> / Tod 1907 `, ` Kopfbedeckung `, ` Dose `, ` Schatulle `, ` Garten `, ` Mystik `, ` Astronomie `, ` Geschirr <Zugtiere> `, ` Farbdruck `, ` Haus / Gebäudeteil `, ` Musiker `, ` Stempel `, ` Zweig `, ` Figur / Frau `, ` Blindpressung `, ` Bambusrohr `, ` Handarbeit `, ` Palmblätter `, ` Abziehbild `, ` Gläser-Set `, ` Korb `, ` Kinderspiel `, ` Fotodruck `, ` König `, ` Schule `, ` Indianer `, ` Keramische Malerei `, ` Gegenwartskunst `, ` Vorratsgefäß `, ` Spielzeug `, ` Bauelement `, ` Karlsruhe `, ` bronziert `, ` Wandteller `, ` Aschenbecher `, ` Tourismus `, ` Umdruckverfahren `, ` Modellierung `, ` Drehen `, ` Damaststahl `, ` gebeizt `, ` Transfrau `, ` Photographie `, ` Brautkleid `, ` Braut `, ` Poster `, ` Propaganda `, ` Fahne `, ` Kolonie `, ` Deutsch-Westafrika `, ` Kamerun <Schutzgebiet> `, ` Andachtsbild `, ` Pieta `, ` Landleben `, ` Geistlicher `, ` Porträtphotographie `, ` Fotokarton `, ` Familienbildnis `, ` Handschrift `, ` Photographie / Gebäude `, ` Landschaftsphotographie `, ` zeitgenössische Fotografie `, ` Identität `, ` Buch `, ` Hinterglasmalerei `, ` Spiegelglas `, ` Emaillieren `, ` Orden <Ehrenzeichen> `, ` Weinlese `, ` Tschechische Republik `, ` Böhmen `, ` Porträt / Frau `, ` Druck `, ` England `, ` Teller `, ` Maria / Jesuskind `, ` Altstadt `, ` Ansichtspostkarte `, ` Zeitschrift `, ` USA `, ` Stillleben `, ` Marktplatz `, ` Mutter / Kind `, ` Kind / Bildnis `, ` Mädchen `, ` Kaffeehaus `, ` Neujahr `, ` Zimmergarnitur `, ` Sitzmöbel `, ` Lithographie `, ` Glaube `, ` Biskuitporzellan `, ` Volkskunst `, ` Fotopostkarte `, ` Nürnberg `, ` Brautkranz `, ` Teppich `, ` Frauenkleidung `, ` Photographie `, ` Schwarzweißphotographie `, ` Park `, ` koloriert `, ` Uniform `, ` Digitalaufnahme `, ` Kopftuch `, ` Dokumentation `, ` Farblithographie `, ` Photoalbum `, ` Comic `, ` glasiert (Seladonglasur) `, ` Lotusblütenmotiv `, ` Figur / Mann `, ` Töpferscheibe `, ` Relief `, ` modelliert `, ` Patina `, ` Eisenglasur `, ` Wein `, ` Eisenblech `, ` Gotik `, ` Aquarell `, ` Deutschland `, ` Alpenlandschaft `, ` Photographie / Personendarstellung `, ` Becher `, ` Haushalt `, ` Wandschmuck `, ` Musik `, ` Kommunismus `, ` Weltkultur `, ` Stoff `, ` Stich <Kunst> `, ` Photographie / Menschen `, ` Hausrat `, ` Wäsche `, ` Krug `, ` Pfeife `, ` Messer `, ` Palmen `, ` Handarbeiten `, ` Ziege `, ` Zigarette `, ` Raucher `, ` Bauarbeit `, ` Schwarzwald / Landschaft `, ` Mütze `, ` Seidenfaden `, ` Tracht `, ` Trinkgefäß `, ` Birkenholz `, ` Tannenholz `, ` Kalender `, ` Dreißigjähriger Krieg `, ` Leopold <Baden, Großherzog, I.> `, ` Bleiruten `, ` Schwarzlot `, ` Glasmalerei `, ` Engel `, ` Trinität `, ` Krieg `, ` Friedrich I., Baden, Großherzog `, ` Breisach am Rhein `, ` Baden (Familie) `, ` Männerkleidung `, ` Karl Friedrich, Baden, Großherzog `, ` Karl Wilhelm III., Baden-Durlach, Markgraf `, ` Durlach `, ` Baden / Wappen `, ` Ofen `, ` Tischlerarbeit `, ` Zunft `, ` Tod `, ` Tischdekoration `, ` Ludwig Wilhelm I., Baden, Markgraf `, ` Junge `, ` Lindenholz `, ` Religiöse Kunst `, ` Jesuskind `, ` Plastik `, ` Mittelalter / Geschichte 1250-1500 `, ` Kupferstich `, ` Prägen `, ` Kästchen `, ` Behälter `, ` Kerze `, ` Heraldik `, ` Ritter `, ` Silbergelb `, ` Konstanz `, ` Reichenau `, ` Allegorie `, ` Gebäckherstellung `, ` Topf `, ` Siedlungsarchäologie `, ` Eheschließung `, ` Apotheke `, ` Arzneimittel `, ` Reformation `, ` Heidelberg `, ` Pfalz `, ` Herakles `, ` Soldat `, ` Zeltlager `, ` Kamin `, ` Lesen `, ` Aktdarstellung `, ` Taufe `, ` Ölfarbe `, ` Kloster `, ` Blattmetallauflagen `, ` Ölfarben `, ` Altar `, ` Himmelfahrt Mariens `, ` Bibel `, ` Winzer `, ` Brunnen `, ` Schifffahrt `, ` Weinrebe `, ` Jahreszeit `, ` Stuhl `, ` Fass `, ` Fuhrwerk `, ` Grabmal `, ` Grabstein `, ` Musikinstrumentenbau `, ` Geschenk `, ` <Baden / Geschichte> `, ` Jubiläum `, ` Universität `, ` Heidelberg / Schloss `, ` Mannheim `, ` Basel `, ` Rhein <Motiv> `, ` Straßburg `, ` Straßburg / Münster `, ` Freiburg im Breisgau `, ` Protestantismus `, ` Trinken `, ` Heilige `, ` Büste / Mann `, ` Trockenplatte `, ` Zimmerpflanze `, ` Wohnzimmer `, ` Töpferei `, ` Arbeit `, ` Schubkarren `, ` Abzug <Photographie> `, ` Töpfer `, ` Werkstatt `, ` Hand `, ` Postkarte `, ` Hausansicht `, ` Atelieraufnahme `, ` Keramiker `, ` Personenphotographie `, ` Holzbildhauer `, ` Albuminpapier `, ` Photographisches Verfahren `, ` Stadt <Motiv> `, ` Schwarzwald `, ` Bollenhut `, ` Schwarzwald <Motiv> `, ` Brautpaar `, ` Kulisse `, ` Atelierphotographie `, ` Trachtengruppe `, ` Tracht <Motiv> `, ` Bauernhaus `, ` Schwarzwaldhaus `, ` Landschaftsaufnahme `, ` Heuwagen `, ` Bäuerliche Familie `, ` Landarbeit `, ` Heuernte `, ` Schmiede `, ` Gasthof `, ` Schwarzwaldlandschaft `, ` Burgruine `, ` Ortsansicht `, ` Hotel `, ` Wallfahrtskirche `, ` Feldberg (Schwarzwald) `, ` Turm `, ` Gasthaus `, ` Fremdenverkehrsort `, ` Bergsee `, ` Tunnel `, ` Eisenbahn `, ` Flusslandschaft `, ` Tal `, ` Höhle `, ` Skulptur `, ` Hirsch `, ` Ruine `, ` Heilbad `, ` Brauerei `, ` Schlucht `, ` Höllental <Schwarzwald> `, ` Viadukt `, ` Innenarchitektur `, ` Thermalbad `, ` PE-Papier `, ` Skisport `, ` Skiläufer `, ` Wandern `, ` Wegweiser `, ` Wald `, ` Ernte `, ` Frauentracht `, ` Fronleichnamsprozession `, ` Fronleichnam `, ` Tor `, ` Rathaus `, ` Kirchenbau `, ` Münster Freiburg (Freiburg im Breisgau) `, ` Pflug `, ` Romanik `, ` Friedhof `, ` Fohlen `, ` Erntedankfest `, ` Brot `, ` Gemüse `, ` Brauchtum `, ` Prozession `, ` Allee `, ` Köhlerei `, ` Feldpostkarte `, ` Weltkrieg <1914-1918> `, ` Photographie / Krieg `, ` Schützengraben `, ` Schlacht `, ` Feldpost `, ` Kriegszerstörung `, ` Befestigung `, ` Fort `, ` Mahlzeit `, ` Motorrad `, ` Kriegsschaden `, ` Markthandel `, ` Weltkrieg <1939-1945> `, ` Gruppenaufnahme `, ` Kanal `, ` Lazarett `, ` Bett `, ` Tanne `, ` Getreideernte `, ` Sense `, ` Trinkglas `, ` Schürze `, ` Schulkind `, ` Brustbild `, ` Kohledruck `, ` Zigeuner `, ` Wegkreuz `, ` Scheune `, ` Allerheiligen `, ` Holzarbeit `, ` Schneflerei `, ` Heimarbeit `, ` Freizeit `, ` Kuh `, ` Schlitten `, ` Frauenarbeit `, ` Photographische Schicht `, ` Negativ <Photographie> `, ` Brenzinger & Cie. (Freiburg im Breisgau) `, ` Baustelle `, ` Öffentliches Gebäude `, ` Fabrikgebäude `, ` Straßenbau `, ` Eisenbahnbrücke `, ` Wasserkraftwerk `, ` Architekturdarstellung `, ` Innenaufnahme `, ` Arbeiter `, ` Schulgebäude `, ` Orgel `, ` Geländer `, ` Visitenkarte `, ` Stausee `, ` Wehr `, ` Säule `, ` Treppe `, ` Kriegerdenkmal `, ` Denkmal `, ` Negativ-Glasplatte `, ` Kraftwerk `, ` Pfeiler `, ` Dia `, ` Stereophotographie `, ` Wasserturm `, ` Krankenhausbau `, ` Ausflug `, ` Bürgertum `, ` Dom `, ` Kapelle `, ` Wallfahrtskapelle `, ` Kanzel `, ` Bürgerhaus `, ` Urlaubsreise `, ` Alpen `, ` Silberhochzeit `, ` Stadt / Platz `, ` Gletscher `, ` Wintersport `, ` Dachstuhl `, ` Portal `, ` Ort `, ` Stadttor `, ` Fachwerkbau `, ` Personenkraftwagen `, ` Grab `, ` Archäologische Stätte `, ` Atelier `, ` Künstler `, ` Kachelofen `, ` Wandmalerei `, ` Dampfschiff `, ` Zoologischer Garten `, ` Bär `, ` Mühle `, ` Bauernhof `, ` Schwein `, ` Kloster / Architektur `, ` Schulklasse `, ` Kruzifix `, ` Stadtmauer `, ` Eingang <Architektur> `, ` Innenhof `, ` Kurort `, ` Schindel `, ` Fastnacht `, ` Maler `, ` Puppe `, ` Villa `, ` Karneval `, ` Steinbruch `, ` Fischerei `, ` Ehepaar `, ` Nähmaschine `, ` Luftschiff `, ` Freibad `, ` Bad `, ` Straße `, ` Stadtbefestigung `, ` Stadtansicht `, ` Baubetrieb `, ` Badekleidung `, ` Badestrand `, ` Insel `, ` Bürgerwehr `, ` Festumzug `, ` Fotograf `, ` Flugzeug `, ` Torbau `, ` Treppenhaus `, ` Natur `, ` Strand `, ` Drittes Reich `, ` Wassersport `, ` Alhambra Granada `, ` Weinbau `, ` Kreuzgang `, ` Hafen `, ` Wochenmarkt `, ` Promenade `, ` Küste `, ` Geschwister `, ` Schlittenfahrt `, ` Segelboot `, ` Brand `, ` Feuerwehr `, ` Bauzeichnung `, ` Zerstörung `, ` Bahnhof `, ` Bahnhofsarchitektur `, ` Stall `, ` Landwirtschaftliches Gebäude `, ` Völkerkunde `, ` Laterna magica `, ` Streich <Scherz> `, ` Druckfarbe `, ` Frühjahr `, ` Heimatverein `, ` Sanatorium `, ` Museumsbau `, ` Bauindustrie `, ` Karte `, ` Ausstellung `, ` Platz `, ` Schäfer `, ` Carte-de-visite `, ` Album `, ` Märchen `, ` Zirkus `, ` Humoristische Darstellung `, ` Veranstaltung `, ` Statue `, ` Mauer `, ` Transportmittel `, ` Film `, ` Wissenschaft `, ` Weltall `, ` Weltraum `, ` Krankenhaus `, ` Zeitungsartikel `, ` Gaststätte `, ` Chorraum `, ` Luftbild `, ` Bergbau `, ` Nacht `, ` Fassade `, ` Röntgenbild `, ` Schlachtfeld `, ` Farbphotographie `, ` Krankheit `, ` Zwerg `, ` Kraftwagen `, ` Hochhaus `, ` Weihnachten `, ` Marktstand `, ` Krankenschwester `, ` Straßenmusikant `, ` Radierung `, ` Kamera `, ` Baden-Württemberg `, ` Schaufenster `, ` Verkäufer `, ` Katholische Kirche `, ` Porträt / Mann `, ` Waisenhaus `, ` Arzt `, ` Familienwappen `, ` Gruppenbildnis `, ` Pfarrer `, ` Gebäudeansicht `, ` Talsperre `, ` Kaschieren `, ` Bibliothek `, ` Stadthalle `, ` Restaurant `, ` Schlosspark `, ` Straßenbahn `, ` Gespann `, ` Film <Material> `, ` Farbdia `, ` Blumenteppich `, ` Kapelle <Musik> `, ` Jahrmarkt `, ` Haube `, ` Gebet `, ` Freilichtmuseum `, ` Erstkommunion `, ` Trachtenfest `, ` Möbelteil `, ` Hase `, ` Thoma, Hans `, ` Auerhuhn `, ` Museum `, ` Schneiderei `, ` Lager <Militär> `, ` Personen/Figuren `, ` Bregger, Emma `, ` Maier, Josef (Töpfer) `, ` Laden `, ` Bregger, Egon `, ` Bürgermeister `, ` Andenken `, ` Druckgraphik `, ` Storch `, ` Haustiere `, ` Stube `, ` Zeitung `, ` Böttcher `, ` Parkplatz `, ` Stricken `, ` Bergbahn `, ` Blumenbeet `, ` Gartenarbeit `, ` Gasthausschild `, ` Spaziergang `, ` Wiese `, ` Schwimmbadbau `, ` Ruderboot `, ` Aussicht `, ` Politik `, ` Historismus `, ` Kirchturm `, ` Verwandtschaft `, ` Schafherde `, ` Kaffeeklatsch `, ` Basteln `, ` Zaun `, ` Turnen `, ` Stadt `, ` Pyramide `, ` Esel `, ` Autochromplatte `, ` Blumenstrauß `, ` Pferdeschlitten `, ` Jugend `, ` Kinderwagen `, ` Helm `, ` Flurdenkmal `, ` Offizier `, ` Klosterkirche `, ` Bildstock `, ` Kaktusgewächse `, ` Grenzstein `, ` Urkunde `, ` Medaille `, ` Gemälde `, ` Männertracht `, ` Zimmerer `, ` Almhütte `, ` Bodensee `, ` Ostern `, ` Springbrunnen `, ` Volkshochschule `, ` Universitätsklinik `, ` Glaube / Katholische Kirche `, ` Comicfigur `, ` Lied `, ` Sehenswürdigkeit `, ` Jäger `, ` Albert-Ludwigs-Universität Freiburg `, ` Verein `, ` Bundesgartenschau `, ` Wassermühle `, ` Graphik `, ` Freizeitpark `, ` Stadtpark `, ` Gymnasium `, ` Spielbank `, ` Bildungseinrichtung `, ` Cartoon `, ` Gottesdienst `, ` Sachsen `, ` Kinderkleidung `, ` Pension <Beherbergungsbetrieb> `, ` Kampagne `, ` Gesundheit `, ` Zeichnung `, ` Satire `, ` Hirt `, ` Viehwirtschaft `, ` Mehrzweckhalle `, ` Panoramakarte `, ` Bach `, ` Grußkarte `, ` Pfingsten `, ` Weihnachtsbaum `, ` Tropfsteinhöhle `, ` Briefträger `, ` Ausstechform `, ` Kirchenfenster `, ` Schüssel `, ` Milchtopf `, ` Tasse `, ` Kaffeekanne `, ` Milchkännchen `, ` Kachel `, ` Salzstreuer `, ` Pfefferstreuer `, ` Blumentopf `, ` Lamm `, ` Untersetzer `, ` Anzug `, ` Rosenkranz `, ` Deutsches Kaiserreich `, ` Baukasten `, ` Hexe `, ` Narr `, ` Brezel `, ` Leiterwagen `, ` Getreide `, ` Kartoffel `, ` Crocus `, ` Musikant `, ` Panorama `, ` Kleinkind `, ` Geburtstag `, ` Kind <Motiv> `, ` Strohschuh `, ` Schnitzhandwerk `, ` Axt `, ` Maurer `, ` Handwerker `, ` Heiligenfigur `, ` Mensch <Motiv> `, ` Schaf `, ` Advent `, ` Adventskranz `, ` Schaukelpferd `, ` Photographie / Gruppe `, ` Igel `, ` Architekturphotographie `, ` Weg `, ` Gitarre `, ` Photographie / Ortsansicht `, ` Pony `, ` Gebäude <Motiv> `, ` Rothirsch `, ` Nachkriegszeit `, ` Radfahrer `, ` Fremdenverkehr `, ` Rind `, ` Belchen `, ` Paar `, ` Bauernstube `, ` Nahrungsaufnahme `, ` Ren `, ` Glockenturm `, ` Schüler `, ` Herde `, ` Konfirmation `, ` Weihnachtsschmuck `, ` Grabplatte `, ` Landstraße `, ` Kriegsschiff `, ` Frankreich `, ` Kriegsschauplatz `, ` Schwarzwaldmädel `, ` Operette `, ` Junger Mann `, ` Napoleon I., Frankreich, Kaiser `, ` Madeira `, ` Photographisches Material `, ` Keramikmanufaktur `, ` Weinberg `, ` Frau / Beruf `, ` Uhrenindustrie `, ` Lastkraftwagen `, ` Wandervogel `, ` Palmprozession `, ` Palmstange `, ` Karwoche `, ` Ministranten `, ` Fastnachtszug `, ` Wagen `, ` Fastnachtskostüm `, ` Schneckenhaus `, ` Kaufhaus / Architektur `, ` Drehorgel `, ` Reisen `, ` Parklandschaft `, ` Tauben `, ` Bank <Möbel> `, ` Tierphotographie `, ` Naturphotographie `, ` Adler `, ` Hausbau `, ` Kutschfahrt `, ` Star <Vogel > `, ` Futterplatz `, ` Negativ `, ` Dahlie `, ` Innenstadt `, ` Hochrhein-Gebiet `, ` Autofahren `, ` Celluloid `, ` Erker `, ` Felsen `, ` Kerzenhalter `, ` Donautal `, ` Torbogen `, ` Spinnrad `, ` Mensch / Tiere `, ` Frauen `, ` Keramikmalerin `, ` Pilz `, ` Sonnenblume `, ` Biene `, ` Silvester `, ` Peitsche `, ` Service <Hausrat> `, ` Villingen `, ` Villingen-Schwenningen / Sankt Ursula / Kloster `, ` Colmar `, ` Schweiz `, ` Narrenzunft `, ` Elsass `, ` Karlsruhe / Badisches Landesmuseum Karlsruhe `, ` Paris `, ` Breisgau `, ` Titisee-Neustadt `, ` Hochschwarzwald `, ` Bayern `, ` Griechenland `, ` Jugoslawien `, ` Heimatmuseum `, ` Volkstanz <Motiv> `, ` Staufen im Breisgau `, ` Deutschland (DDR) `, ` Siebenbürgen `, ` Oberkirch <Ortenaukreis> `, ` Verkehr `, ` Verkauf `, ` Strumpf `, ` Fußbekleidung `, ` Stadtteil `, ` Endingen am Kaiserstuhl `, ` Kaiserstuhl `, ` Verkehrsmittel `, ` Post `, ` Preußen `, ` Berlin `, ` Bernau im Schwarzwald `, ` München `, ` Kur `, ` Karlsruhe / Schloss `, ` Rhein `, ` Badenweiler `, ` Hessen `, ` Pforzheim `, ` Albtal <Sankt Blasien> `, ` Elektrifizierung `, ` Stuttgart `, ` Landkreis Breisgau-Hochschwarzwald `, ` Titisee `, ` Bodensee-Gebiet `, ` Hus, Jan `, ` Unterkunft `, ` Wiesental `, ` Bad Wildbad `, ` Beuron / Kloster `, ` Beuron `, ` Rumänien `, ` Rheinfall `, ` Säckingen `, ` Brief `, ` Markgräfler Land `, ` Abend `, ` Meersburg `, ` Kopf / Frau `, ` Kopf / Mann `, ` Geburt Jesu `, ` Offenburg `, ` Ehe `, ` Renaissancearchitektur `, ` Hochaltar `, ` Feldberg `, ` Gengenbach `, ` Lörrach `, ` Baden `, ` Basilika `, ` Rastatt `, ` Botanischer Garten `, ` Hebel, Johann Peter `, ` Europa `, ` Württemberg `, ` Fachwerk `, ` Wien `, ` Bauernmöbel `, ` Benediktiner `, ` Branntweinflasche `, ` Pfarrkirche `, ` Müllheim `, ` Papst `, ` Holzplastik `, ` Postkutsche `, ` Blumenschmuck `, ` Baden-Baden `, ` Kaufhaus `, ` Donau `, ` Hirsau `, ` Gebäude `, ` Mainau `, ` Werbemittel `, ` Evangelische Kirche `, ` Vogesen `, ` Schiller, Friedrich `, ` Johannes <der Täufer> `, ` Lehrer `, ` Modell `, ` Berg <Motiv> `, ` Tiefdruck `, ` Sparkasse `, ` Karikatur `, ` Schwabentor Freiburg im Breisgau `, ` Kurpark `, ` Schlafzimmer `, ` Spruch `, ` Werbeplakat `, ` Kurhaus `, ` Ölmalerei `, ` Darstellung `, ` Schottland `, ` Schwarzwald <Süd> `, ` Aussichtsturm `, ` Schloss / Gebäudeteil `, ` Baden <Land> `, ` Volkskunde `, ` Hof `, ` Dampflokomotive `, ` Demonstration `, ` Tisch `, ` Geschäftshaus `, ` Umzug `, ` Schlossberg Freiburg im Breisgau `, ` Wand `, ` Sanierung `, ` Diapositiv (farbig) `, ` Staufen / Keramikmuseum `, ` Ladeneinrichtung `, ` Colmar <Elsass> `, ` Filmdokumentation `, ` Mittelalter `, ` Freiburg / Lorettoberg `, ` Wasserburg `, ` Kopf `, ` Heu `, ` Sänger `, ` Mundart `, ` Abenteuerroman `, ` Karussell `, ` Freundschaft `, ` Südbaden `, ` Plakat `, ` General `, ` Auszeichnung `, ` Spiel / Utensil `, ` Schwarzweißdia `, ` Milchproduktion `, ` Karren `, ` Walmdach `, ` Urlaub `, ` Schulbuch `, ` Teeservice `, ` Sakralbau `, ` Hochschulbau `, ` PE-Fotopapier `, ` Schiffsreise `, ` Kultur `, ` Gastronomie `, ` Speisezimmer `, ` Dokument `, ` Schauinsland `, ` Oldtimer `, ` Laterne `, ` Gutacher Malerkolonie `, ` Nikolaus <Heiliger> `, ` Technische Zeichnung `, ` Schreibtisch `, ` Büro `, ` Russland `, ` Feier `, ` Belgien `, ` Oudenaarde `, ` Brüssel `, ` Brügge `, ` Kortrijk `, ` Gent `, ` Tournai `, ` Militärflugzeug `, ` Rom `, ` Florenz `, ` Luxor `, ` Reproduktionsphotographie `, ` Weinflasche `, ` Marine `, ` Technik `, ` Bruchsal `, ` Pflasterstein `, ` Religion/Mythologie/Kult `, ` Hilda <Baden, Großherzogin> `, ` Friedrich <Baden, Großherzog, II.> `, ` kopiert `, ` Cabinet `, ` Epitaph `, ` Feldweg `, ` Tortenplatte `, ` Stuhllehne `, ` Kaffeegeschirr `, ` Maskerade `, ` Vorhang `, ` Großmutter `, ` Rotes Kreuz `, ` Kirchenfest `, ` Silber-Gelatine-Abzug `, ` Tapete `, ` Plattenspieler `, ` Presse `, ` Kindergarten `, ` Politiker `, ` Kartonage `, ` Glasmalerei `, ` Luise, Baden, Großherzogin `, ` Wilhelm I., Deutsches Reich, Kaiser `, ` Diapositiv (schwarzweiß) `, ` Fotocollage `, ` Brettstuhl `, ` Germanen `, ` Baudekoration `, ` Biblische Darstellung `, ` Feldküche `, ` Staatliches Museum für deutsche Volkskunde `, ` Gondel <Boot> `, ` Wachspuppe `, ` Wettkampf `, ` Sportveranstaltung `, ` Rennfahrer `, ` Motorradrennmaschine `, ` Wildschwein `, ` Volkstanz `, ` Luther, Martin `, ` Kriegsgefangenschaft `, ` Hühnerhaltung `, ` Gebirgszug `, ` Mozart, Wolfgang Amadeus `, ` Karlsruher Schloss `, ` Bernhard <Baden, Markgraf, I.> `, ` Marktfrau `, ` Gebäck `, ` Kolorierung `, ` Erziehung `, ` Ebene `, ` Steinmetz `, ` Stuck `, ` Gäste `, ` Ratsche <Musikinstrument> `, ` Drama `, ` Klavier `, ` Schlafmütze `, ` Nachthemd `, ` Lichtdruck `, ` Vorlage `, ` Sattel `, ` Tagebuch `, ` Amateurphotographie `, ` Photographische Platte `, ` T-Shirt `, ` Feldforschung `, ` Film 16 mm `, ` Kragen `, ` Hochrhein `, ` Antikernkraftbewegung `, ` Revolution <1848> `, ` Protest `, ` Stadtbild `, ` Kinderbuch `, ` Ungarn `, ` Afrika `, ` Fotozubehör `, ` Federlithographie `, ` Kreidelithographie `, ` Kostümkunde `, ` Papiertheater `, ` Typographie `, ` Guckkasten `, ` Schießsport `, ` Deutsches Reich `, ` Dänemark `, ` Glücksspiel `, ` Moral `, ` Sinnspruch `, ` Schweden `, ` Holzstich `, ` Brüder Grimm `, ` Bühnenbild `, ` Prägedruck `, ` Boulevardtheater `, ` Komödie `, ` <Frankreich / Geschichte> `, ` Schießscheibe `, ` Bilderbogen `, ` Wissensvermittlung `, ` Afrikaner `, ` Modellbogen `, ` Offsetdruck `, ` Wilder Westen `, ` Flexodruck `, ` handkoloriert `, ` Musikalien `, ` Bild im Rahmen `, ` Religiosität `, ` Shutdown `, ` Quarantäne `, ` COVID-19 (Coronavirus SARS-CoV-2) `, ` Wässrige Lösung `, ` Abfüllung `, ` Gemisch `, ` social distancing `, ` Hygieneartikel `, ` Gummiband `, ` Mensch-Ding-Verflechtung `, ` Atem `, ` Gesicht `, ` Gesichtsmaske `, ` Prävention `, ` Fertigung `, ` Näharbeit `, ` Öffnung `, ` Barriere `, ` Vinyl `, ` Tonträger `, ` Broschüre `, ` Stoffdruck `, ` Formen `, ` Merchandising `, ` Nahrung `, ` Take-away `, ` Drahtrückstichheftung `, ` Studium `, ` handbemalt `, ` Klebebindung `, ` Hoffnung `, ` Gesundheitspolitik `, ` Karlsruhe / Schlossanlage `, ` Serviette `, ` Feinmechaniker `, ` Goldene Hochzeit `, ` Friedrich <Baden, Großherzog, I.> / Jubiläum <1906> `, ` Schallplatte `, ` Schlager `, ` Barrytpapier `, ` Handabzug <Photographie> `, ` Paperback `, ` Unterhaltungsmusik `, ` Zubehör `, ` Amateurfilm `, ` Eckert, Georg Maria `, ` Musikanlage `, ` Silberhalogenidsalz `, ` Italiensouvenir `, ` Acella `, ` Stereobild `, ` Popmusik `, ` Abspielgerät `, ` Freikörperkultur `, ` Biographie `, ` Schmidt-Staub (Familie) `, ` Friedrich <Baden, Großherzog, I.> / Jubiläum <1896> `, ` Photographie / Natur `, ` Isbjörn (Schiffsname) `, ` Österreichisch-ungarische Nordpolexpedition <1872-1874> `, ` Wilczek, Johann Nepomuk, Graf von `, ` Arktis `, ` Expedition `, ` Spitzbergen `, ` Fototechnik `, ` Schachtel `, ` Feinguss `, ` Irdengut / Scherben / Gelb `, ` Bauplastik `, ` Scherben <keramischer Werkstoff> `, ` Sehne `, ` Figurenautomat `, ` Musikautomat `, ` Kirschbaumholz `, ` Musikmöbel `, ` Mechanisches Musikinstrument `, ` Spieldose `, ` Papiernotenrolle `, ` Walzen `, ` Blechnerei `, ` Pianola `, ` Lochplatte `, ` Musikdose `, ` Player Piano `, ` Notenrolle `, ` Ptolemäer `, ` Sizilien `, ` Billon `, ` Probemünze `, ` Stempelsiegel `, ` Friedrich <Baden, Großherzog, I.> / Jubiläum <1902> `, ` Notgeld `, ` Inflation `, ` Banknote `, ` Heft `, ` Kamee `, ` Gemme `, ` Währungseinheit `, ` Nickel `, ` Büste / Frau `, ` Rastatt / Friede `, ` Spanischer Erbfolgekrieg `, ` Tetradrachmon `, ` Kreuzer <Münze> `, ` Taler `, ` Gedenkmünze `, ` Tombak `, ` Rohling <Fertigung> `, ` Deutsche Mark `, ` Orden von Zähringer Löwen `, ` Euro <Währung> `, ` Raumfahrt `, ` Bronzemünze `, ` Vergoldetes Messing `, ` Neusilber `, ` Flugzeugführer `, ` Notarbeit `, ` Umdruckdekor `, ` Bleiglas `, ` Majolika `, ` Keramik <Deutschland> `, ` Palisanderholz `, ` Steingut / Scherben / Weiß `, ` Betriebsverpflegung `, ` Kantine `, ` Rüsterholz `, ` Craqueléglasur `, ` angarniert `, ` Studienarbeit `, ` Uranglasur `, ` Kristallglasur `, ` Schamotte `, ` Feinsteinzeug `, ` aufgebaut `, ` Künstlerspielzeug `, ` Ascheglasur `, ` Seidenmattglasur `, ` Keramik <1960er Jahre> `, ` Mattglasur `, ` Abstrakte Plastik `, ` Lehmglasur `, ` Feldspatglasur `, ` Rostglasur `, ` Jugendstil <Österreich> `, ` Revolutionsporzellan `, ` Kobaltglasur `, ` Eisendraht `, ` Farbglasur `, ` ausgeformt `, ` Schamotteton `, ` Lampenglas `, ` Metallfolie `, ` Venedig-Murano `, ` Bleikristall `, ` Pressglas `, ` Beschichtung `, ` Cromargan `, ` Abstrakte Kunst `, ` Styropor `, ` Selenglasur `, ` Fat Lava `, ` Kraterglasur `, ` Sterlingsilber `, ` Küchenmöbel `, ` Siedlungsbau `, ` Hotelgeschirr `, ` Stapelgeschirr `, ` Systemgeschirr `, ` Reduktionsbrand `, ` Transparentglasur `, ` Kupferglasur `, ` Kugel `, ` Ölfleckenglasur `, ` gestrickt `, ` Irdengut / Scherben / weiß `, ` Irdengut / Scherben / rötlich-braun `, ` Eierbecher `, ` Karlsruhe / Siedlung Dammerstock `, ` Set <Hausrat> `, ` gespritzt `, ` Deckeldose `, ` Zuckerdose `, ` Glasware `, ` Gedeck `, ` eloxiert `, ` Umdruck `, ` Band <Textilien> `, ` Trendobjekt <2010er Jahre> `, ` Kork `, ` Spirale `, ` Einbauküche `, ` Dammerstock `, ` Vierfarbdruck `, ` Westafrika `, ` Sammlungsgeschichte `, ` Filmprogramm `, ` Bouillon `, ` Glaselemente `, ` Lamé `, ` Getragenes Accessoire `, ` Oberkleidung `, ` Hemd `, ` gefeilt `, ` Schnalle `, ` Halstuch `, ` Alkoholisches Getränk `, ` Kaffeebohne `, ` Lahn `, ` geschneidert `, ` Moiréseide `, ` Wolltuch `, ` Figurine `, ` Ausstellungstechnik `, ` Fadenheftung `, ` Broschur `, ` Halbgewebeband `, ` Ganzgewebeband `, ` Drahtseitstichheftung `, ` Pappband `, ` Humoristische Literatur `, ` Kordel <Schnur> `, ` Eindrehen `, ` Überformen `, ` Anbietschale `, ` Saftkrug `, ` Sammeltasse `, ` Biegen `, ` Ständer `, ` Anbietgefäß `, ` Cognacschwenker `, ` Alkoholfreies Getränk `, ` Messingdraht `, ` Illustrierte `, ` Hardcover `, ` Metallklammern `, ` Ratgeber `, ` Politische Literatur `, ` Uhrwerk `, ` Cocktailspieß `, ` Resopal `, ` Serviettenhalter `, ` Prospekt `, ` Flaschenöffner `, ` Krakeleeglas `, ` Romanheft `, ` Jugendbuch `, ` Roman `, ` Streichhölzer `, ` Sachbuch `, ` Cocktailrezepte `, ` Kriminalgeschichte `, ` Ausschneidebogen `, ` Schmucknadel `, ` Kundenzeitschrift `, ` Produktverpackung `, ` Pop-up-Technik `, ` Rauchglasur `, ` Terra sigillata `, ` Türkenbeute / Karlsruhe / Badisches Landesmuseum Karlsruhe `, ` Pferdezaumzeug `, ` Janitscharen `, ` Vergoldeter Silberlahn `, ` Vergoldeter Silberdraht `, ` Vergoldetes Silberblech `, ` not colourful `, ` slightly colourful `, ` moderately colourful `, ` averagely colourful `,  ` quite colourful `, ` highly colourful `, ` extremely colourful `


## Business


### Costs

**Pong** can operate very cost-effectively, since there is no need for any cloud service to run the system. The web engine can run on a common web server. We already trained the AutoEncoder networks, but if there is a need to retrain, it can be trained on a common GPU or CPU. Adding new artwork to the system is easy, since the one-hot encoding process can be made easily based on museum data. Creating the target vector by AIs can be made on CPU or common GPU. 


### Output data

The demo or the models do not collect or store personal or user data. However, it is possible to collect and store telemetry data like ratings, device identifiers or user data, if there is need. In this case the data can help museums to monitor the taste of the audience, or it can be the base of a scientific research or PhD dissertation. 


## Demo

Link: ` private ` - due to the legal limitation of the original test dataset

The demo page is tested with Firefox or Chrome. It recommends **27.898** artworks 🖼 based on **2074** keywords 💬, **7** different colorfulness categories 🌈 or outputs of AI 🧠 or ML 🗜 models.


![Demo 1](https://github.com/hyperrixel/pong/blob/main/asset/demo_1.png "Demo 1 - main page")
![Demo 2](https://github.com/hyperrixel/pong/blob/main/asset/demo_2.png "Demo 2 - help")


## Future

In the future we would like to make experiments with other rating methods. We are really curious about a recommendation based on real emotions or a recommendation where the rating is based on verbal feedback only. This last one means the integration of an ASR and an NLP module. 


