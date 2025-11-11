library(tidyverse)
library(dplyr)

data()

View(mpg)

colnames(mpg)

glimpse(mpg)

filter(mpg, cty >= 15)
mpg_normal_useable <- filter(mpg, cty >= 15)

mpg_ford <- filter(mpg, manufacturer == "ford")
View(mpg_ford)

mpg_metric <- mutate(mpg, cty_metric = 0.425144 * cty)
View(mpg_metric)

mpg_metric <- mpg %>%
  mutate(cty_metric = 0.425144 * cty)

View(mpg_metric)

View(mpg)

mpg %>%
  group_by(class) %>%
  summarize(mean(cty),
            median(cty))

# summarize - US English | summarise - UK English

# Data viz with ggplot2

ggplot(mpg, aes(x = cty)) +
  geom_histogram() +
  labs(x = "City Mileage")

ggplot(mpg, aes(x = cty)) +
  geom_freqpoly() +
  labs(x = "City Mileage")

ggplot(mpg, aes(x = cty)) +
  geom_histogram() +
  geom_freqpoly() +
  labs(x = "City Mileage")

ggplot(mpg, aes(x = cty, 
                y = hwy)) +
  geom_point() +
  geom_smooth(method = "lm")

ggplot(mpg, aes(x = cty, 
                y = hwy,
                color = class)) +
  geom_point() +
  scale_color_brewer(palette = "Dark2")



