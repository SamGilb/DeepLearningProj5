
#Code setup
library(tensorflow)
library(keras)
library(ggplot2)
library(data.table)
set.seed(1)

###############################################################################
###############################################################################
###################### DATA UPLOAD AND ORGANIZATION ###########################
###############################################################################
###############################################################################

## Spam Upload
if(!file.exists("spam.data"))
{
  download.file( 
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data", "spam.data")
}

spam.dt <- data.table::fread("spam.data")
## Need to convert Spam DT to an array
label.col <- ncol(spam.dt)
Y.Arr <- array( spam.dt[[label.col]], nrow(spam.dt) )

fold.vec <- sample(rep(1:5, l=nrow(spam.dt)))
test.fold <- 1
is.test <- fold.vec == test.fold
is.train <- !is.test

## Scale Data (X)
X.sc <- scale(spam.dt[, -label.col, with=FALSE])
X.train.mat <- X.sc[is.train, ]
X.test.mat <- X.sc[is.test, ]
# Matrices to Arrays, Possibly not necessary
X.train.a <- array(X.train.mat, dim(X.train.mat))
X.test.a <- array(X.test.mat, dim(X.test.mat))

## Set up Y.train and Y.test
Y.train <- Y.Arr[is.train]
Y.test <- Y.Arr[is.test]

###############################################################################
###############################################################################
############################# Train Model #####################################
###############################################################################
###############################################################################

#hyperparameters
n.hyperparameters <- 25
n.neurons.vec <- 5 * 1:n.hyperparameters
n.epochs <- 100

#data storage variables
train.loss.list <- list()
val.loss.list <- list()
metrics.list <- list()
min.metrics.list <- list()


for( curr.hp in 1:n.hyperparameters )
{
  #Initialize model with architecture of (ncol(X), 10, 1)
  model <- keras_model_sequential() %>% 
    layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
    layer_dense(units = n.neurons.vec[curr.hp], activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
    layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer
  
  ## Compile Model for binary classification
  model %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  
  ## Fit model here
  result <- model %>% 
    fit(
      x = X.train.mat, y = Y.train,
      epochs = n.epochs,
      validation_split = 0.5, #0.5 means 50% validation data
      verbose = 2
    )
  
  #define metrics for later use
  metrics <- do.call(data.table, result$metrics)
  metrics[, epoch := 1:.N]
  #define the metrics where validation loss is minimized
  min.metrics <- metrics[which.min(val_loss)]
  best_epoch <- min.metrics$epoch
  min.val_loss <- min.metrics$val_loss
  
  #store metrics for later use in list
  metrics.list[[curr.hp]] <- metrics
  min.metrics.list[[curr.hp]] <- min.metrics
  
  train.loss.list[[curr.hp]] <- metrics$loss[n.epochs]
  val.loss.list[[curr.hp]] <- metrics$val_loss[n.epochs]
  
}

###############################################################################
###############################################################################
################################# Plotting ####################################
###############################################################################
###############################################################################


plots <- list()
#initialize ggplot
p <-ggplot()

for( curr.hp in 1:n.hyperparameters )
{
  n.neurons <- sprintf( "%d Hidden Units", n.neurons.vec[curr.hp])
  #add train loss to plot
  p = p + geom_line(aes(
    x=epoch, y=loss, color = !!n.neurons, linetype = 'train'),
    data = metrics.list[[curr.hp]])
  
  #add validation loss to plot
  p = p + geom_line(aes(
    x=epoch, y=val_loss, color = !!n.neurons, linetype = 'validation'),
    data = metrics.list[[curr.hp]])
}
#append to plot list for possible later use and display
plots[[1]] <- p
print(plots[[1]])

###############################################################################
######################### Plot Second Graph ###################################
###############################################################################

#initialize data for ggplot
train.data <- data.table( unlist(train.loss.list), unlist(n.neurons.vec))
val.data <- data.table( unlist(val.loss.list), unlist(n.neurons.vec))
setnames(train.data, "V1", "Loss")
setnames(train.data, "V2", "Hidden_Units")
setnames(val.data, "V1", "Loss")
setnames(val.data, "V2", "Hidden_Units")
train.min <- train.data[ which.min(train.data$Loss)]
val.min <- val.data[which.min(val.data$Loss)]

#initialize second plot
p2 <- ggplot()
  
#add the training loss
  p2 = p2 + geom_line(aes(
    x=Hidden_Units, y=Loss, color = 'train', linetype = 'train'),
    data = train.data)
  
#add the validation loss
  p2 = p2 + geom_line(aes(
    x=Hidden_Units, y=Loss, color = 'validation', linetype = 'validation'),
    data = val.data)

  p2 = p2 + geom_point(aes(
    x= Hidden_Units, y=Loss, color = 'min'),
    data = train.min)

  p2 = p2 + geom_point(aes(
    x = Hidden_Units, y= Loss, color = 'min'),
    data = val.min)
  
#append to plot list for possible later use and display
plots[[2]] <- p2
print(plots[[2]])

###############################################################################
###############################################################################
############################# Re-Train Model ##################################
###############################################################################
###############################################################################

#hyperparameters
best_parameter_value <- val.min$Hidden_Units

  #Initialize model with architecture of (ncol(X), 10, 1)
  model <- keras_model_sequential() %>% 
    layer_flatten(input_shape = ncol(X.train.mat)) %>% #Input layer
    layer_dense(units = best_parameter_value, activation = "sigmoid", use_bias = FALSE) %>% # Hidden layer, change # of hidden units here? (10, 100, 1000)
    layer_dense(units = 1, activation = "sigmoid", use_bias = FALSE) # Output Layer
  
  ## Compile Model for binary classification
  model %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  
  ## Fit model here
  result <- model %>% 
    fit(
      x = X.train.mat, y = Y.train,
      epochs = n.epochs,
      validation_split = 0, #0 means no validation data
      verbose = 2
    )
  
  evaluation <- model %>%
    evaluate( X.test.mat, Y.test, verbose = 0)
  
print(c("The models final accuracy is:", evaluation$accuracy))
print(c("Baseline accuracy is:", max( sum(Y.test==1)/length(Y.test), sum(Y.test==0)/length(Y.test))))
