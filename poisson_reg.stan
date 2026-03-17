data {
  int<lower=1> N;            // # of orders
  array[N] int<lower=0> y;   // cart_size (count), array syntax required
  vector[N] x1;              // order_number
  vector[N] x2;              // order_dow
  vector[N] x3;              // order_hour_of_day
  vector[N] x4;              // avg_cart_size
}
parameters {
  real beta0;
  real beta1;
  real beta2;
  real beta3;
  real beta4;
}
model {
  // Priors (on log-scale)
  beta0 ~ normal(0,1);
  beta1 ~ normal(0,1);
  beta2 ~ normal(0,1);
  beta3 ~ normal(0,1);
  beta4 ~ normal(0,1);

  // Likelihood
  for (n in 1:N) {
    real eta = beta0
               + beta1 * x1[n]
               + beta2 * x2[n]
               + beta3 * x3[n]
               + beta4 * x4[n];
    y[n] ~ poisson_log(eta);
  }
}
