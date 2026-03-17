data {
  int<lower=1> N;                 // # observations
  int<lower=1> U;                 // # users
  int<lower=1> P;                 // # products
  array[N] int<lower=1,upper=U> user_idx;
  array[N] int<lower=1,upper=P> prod_idx;
  vector[N] ord_num;
  vector[N] ord_dow;
  vector[N] ord_hour;
  array[N] int<lower=0,upper=1> y;
}
parameters {
  real alpha;                     
  real<lower=0> sigma_u;          
  real<lower=0> sigma_p;          
  vector[U] u_raw;                
  vector[P] p_raw;                
  real beta_num;                  
  real beta_dow;                  
  real beta_hour;                 
}
transformed parameters {
  vector[U] gamma_u = u_raw * sigma_u;
  vector[P] delta_p = p_raw * sigma_p;
}
model {
  // Priors
  alpha     ~ normal(0,1);
  sigma_u   ~ normal(0,1);
  sigma_p   ~ normal(0,1);
  u_raw     ~ normal(0,1);
  p_raw     ~ normal(0,1);
  beta_num  ~ normal(0,1);
  beta_dow  ~ normal(0,1);
  beta_hour ~ normal(0,1);

  // Likelihood
  for (n in 1:N) {
    real eta = alpha
               + gamma_u[user_idx[n]]
               + delta_p[prod_idx[n]]
               + beta_num  * ord_num[n]
               + beta_dow  * ord_dow[n]
               + beta_hour * ord_hour[n];
    y[n] ~ bernoulli_logit(eta);
  }
}
