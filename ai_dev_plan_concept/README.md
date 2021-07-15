# The concept of the development

We want to create a recommendation system based on the Museum's data. First of all, we want to use the existing metadata, but a good solution is future-proof, so we have an idea to handle **DAM** data. 
Data is collected from **4** different source:
- Metadata Digital Objects
- DAM Metadata
- Pong data (made by scripts)
- User data (collected directly from the user)
User data is the target of the AI or ML system. Two kinds of data is handled: ratings and actions. Action is any other user feedback than it is not a rating, for example when they closes the browser's tab or rejects to rate the artwork, or the time length of the rating. Anything can be an action based on the experiences of problem understanding. If the professional staff says, this is not relevant, it is easily removable from the concept.

# Development roadmap

## Problem

### Recommendation

1. Check how work the real-life recommendation systems
2. Understanding what is the difference between recommend an artwork and a song, a book or  a movie.
3. Understanding what is the motivation of people to see art
4. Talk with the Museum's staff to find out which data is relevant for choosing an artwork based on their professional experiences

### Software system

1. Talk about the expected outcomes of our system (should we provide only the asset identifier to the museum's server or should we write the whole framework to collect the data and provide the recommendation?)
2. Which is the expected readiness level (TRL) till the end of the hackathon?
3. Which is the expected readiness level (TRL) at the end of the AI development phase?

### Legal

1. The licence of code must be open source (eg. under MIT)

## Data

### Exploring

1. Understand the meaning and contents of data records
2. Process Museum's data to find how many data points (records) can be generated from them
3. Name the bottleneck(s) or pain points in the data: which data is poor

### Modeling

1. Finalize the data model structure based on the experience of problem understanding and the richness or poorness of data

![Datastructure](https://github.com/hyperrixel/pong/blob/main/ai_dev_plan_concept/datastructure.jpg "Datastructure")

### Preprocessing

1. Create basic scripts to transform the given data into our data model
2. Cleaning the data

## Build an AI

### Identify tasks and AI/ML structure

1. To work with textual databese elements, like descriptions, titles, etc. Natural Language Processing is needed.
2. To work with with texts from a categorical aspect one hot encoding and therefore embedding is needed.
3. To work with imaging input Computer Vision is needed.
4. To extract information that is hidden in a more simple way Machine Learning is needed.
5. To extract information that is hidden in a more sophisticated way Neural Networks, Deep Learning is needed.

### Inputs and targets

1. Data must be transformed into potential inputs for models. Key activites of this transformation is to create flexibel categories and adequate normalization.
2. Need to identify targets with the sources to be able to make a supervised learning.

### Selecting framework

## Testing

### Integration testing

### Alpha testing

## Integrating

### Real integration

### Beta testing
