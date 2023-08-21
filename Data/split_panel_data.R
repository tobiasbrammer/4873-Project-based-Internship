split_panel_data <- function(data, id_variable, time_variable, train_prop = 0.6, val_prop = 0.2) {

  # List of unique entities
  entities <- unique(data[[id_variable]])

  # Shuffle the entities
  set.seed(123) # for reproducibility
  entities <- sample(entities)

  # Calculate the number of entities for each set
  n_entities <- length(entities)
  train_n <- floor(n_entities * train_prop)
  val_n <- floor(n_entities * val_prop)

  # Split the entities
  train_entities <- entities[1:train_n]
  val_entities <- entities[(train_n + 1):(train_n + val_n)]
  test_entities <- entities[(train_n + val_n + 1):n_entities]

  # Split the data based on entities
  train_data <- data[data[[id_variable]] %in% train_entities, ]
  val_data <- data[data[[id_variable]] %in% val_entities, ]
  test_data <- data[data[[id_variable]] %in% test_entities, ]

  list(train = train_data, validation = val_data, test = test_data)
}

# Example usage:
# Assuming `panel_data` is your panel dataset with `entity_id` as the entity identifier and `year` as the time variable
# sets <- split_panel_data(panel_data, id_variable = "entity_id", time_variable = "year")
# train_set <- sets$train
# val_set <- sets$validation
# test_set <- sets$test
