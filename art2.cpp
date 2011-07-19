
#include<iostream>
#include <stdlib.h>
#include <stdarg.h>


using namespace std;

art2::art2(float a, float b, float c, float d, float theta, float rho)
{
  A = a;
  B = b;
  C = c;
  D = d;
  theta = theta;
  rho = rho;
}

void art2::set_art2_parameters(float a, float b, float c, float d, float theta, float rho)
{
  A = a;
  B = b;
  C = c;
  D = d;
  theta = theta;
  rho = rho;
}

class layer *art2::build_layer (int units)
{
  class layer *l;

  l = (class layer *) calloc (1, sizeof(layer));
  l->units = units;
  l->inputs = 0;
  l->modifier = 0.0;
  l->initval = 1.0;
  l->outputs = (float *) calloc (units, sizeof(float));
  l->connects = NULL;
  l->activation = (afn) linear;       /* default activation function */
  l->propto = (pfn) dot_product;      /* default propagation function */

  #ifdef BPN
   l->lastdelta = NULL;
   l->errors = (float *) calloc (units, sizeof(float));
   l->deriv = (afn) linear_derivative;
   l->eta = 0.5;
   l->alpha = 0.0;
  #endif

  return (l);
}

class layer** art2::build_net (int layers, int *sizes)
{
  class layer **n;
  register int i;

  n = (layer **) calloc (layers, sizeof(layer *));

  for (i=0; i<layers; i++)
   n[i] = build_layer (sizes[i]);

  return (n);
}

int* art2::define_layers (int layers, ...)
{
  va_list argptr;
  int *l, i;

  l = (int *) calloc (layers+1, sizeof(int));
  l[0] = layers;

  va_start (argptr, layers);
  for (i=1; i<=layers; i++) l[i] = va_arg (argptr, int);
  va_end (argptr);

  return (l);
}

void art2::build_art2 (int *sizes)
{
  int i, layers;

  layers = sizes[0] + 1;
  sizes[0] = sizes[1];      /* Input layer will be same size as F1 */
  net = build_net (layers, sizes);
  layers = layers;

  f1.w = (float *) calloc (sizes[1], sizeof(float));
  f1.x = (float *) calloc (sizes[1], sizeof(float));
  f1.v = (float *) calloc (sizes[1], sizeof(float));
  f1.u = (float *) calloc (sizes[1], sizeof(float));
  f1.p = (float *) calloc (sizes[1], sizeof(float));
  f1.q = (float *) calloc (sizes[1], sizeof(float));
  f1.r = (float *) calloc (sizes[1], sizeof(float));

  patterns = (iop *) calloc (1, sizeof(iop));
  patterns->dimX = -1;   /* no exemplars yet */
  patterns->dimY = -1;
  patterns->invecs = NULL;
  patterns->outvecs = NULL;

  exemplars = 0;
  strcpy (filename, "");
  free (sizes);

  return (n);
}

void art2::connect_layers (class layer *inlayer, class layer *tolayer)
{
  register int i;

  tolayer->inputs = inlayer->units;
  tolayer->connects = (float **) calloc (tolayer->units, sizeof(float *));

  for (i=0; i<tolayer->units; i++)
   tolayer->connects[i] = (float *) calloc (tolayer->inputs, sizeof (float));

  #ifdef BPN
   tolayer->lastdelta = (float **) calloc (tolayer->units, sizeof(float *));

   for (i=0; i<tolayer->units; i++)
     tolayer->lastdelta[i] = (float *) calloc (tolayer->inputs, sizeof(float *));
  #endif
}

void art2::set_activation (class layer *l, afn activation, float modifier)
{
  l->activation = activation;
  l->modifier = modifier;

  #ifdef BPN
    if (activation == SIGMOID)  l->deriv = (afn) sigmoid_derivative;
    if (activation == LINEAR)   l->deriv = (afn) linear_derivative;
  #endif
}

void art2::set_propagation (class layer *l, pfn netx)
{
  l->propto = netx;
  set_activation (l, LINEAR, 1.0);    /* reset to default */
}

void art2::connect (class layer *inlayer, class layer *tolayer, int how, int init)
{
  if (how == COMPLETE)
   {
     connect_layers (inlayer, tolayer);
     set_propagation (tolayer, DOT_PRODUCT);
   }
  else
   if (how == ONE_TO_ONE) set_propagation (tolayer, TRANSFER);

  set_weights (tolayer, init);
}

int art2::train_net (char *filename)
{
  int i, j, pattern, winner;
  float *p, degree_of_match, **savewts, *zeros;

  strcpy (n->filename, filename);
  exemplars = load_exemplars (filename, patterns);

  if (!valid_exemplars (n))
   {
    cout << "\NError: Exemplars do not match network size!" << endl;
    exit (0);
   }

  savewts = (float **) calloc (F2->units, sizeof (float *));
  zeros = (float *) calloc (F1->units, sizeof (float));

  for (i=0; i<F1->units; i++) zeros[i] = 0.0;

  for (i=0; i<2; i++)
   for (pattern=0; pattern<exemplars; pattern++)
    {
     for (j=0; j<F2->units; j++) savewts[j] = F2->connects[j];

     for (;;)
      {
       apply_input (n, get_invec (n, pattern));
       prop_through (n);
       degree_of_match = compare_patterns (n);

       if (degree_of_match <= 1.0) break;

       winner = F2->processed;
       inhibit (winner, F2, zeros);
      }

     for (j=0; j<F2->units; j++) F2->connects[j] = savewts[j];

     adjust_bottom_up_weights (n);
     adjust_top_down_weights (n);
    }

  free (savewts);
  free (zeros);
  return (TRUE);
 }

int main(void)
{

  art2 *n = new art2(10, 10, 0.1, 0.9, 0.2, 0.9);
  int *layers = n->define_layers (2, 3, 4);
  n->build_art2 (layers);
  n->set_art2_parameters (10, 10, 0.1, 0.9, 0.2, 0.9);

  //F1->initval = 0.0;    /* top down weights are initialized to zero */
  //F2->initval = 1.0 / ((1.0 - n.D) * sqrt ((float) F1->units));

  connect (F1, F2, COMPLETE, VALUE);
  connect (F2, F1, COMPLETE, VALUE);

  set_parameters (F1, DOT_PRODUCT, LINEAR, 1.0);
  set_parameters (F2, DOT_PRODUCT, ON_CENTER, 0.0);

  if (train_net (n, "art2test.dat")) show_net (n);  /* save_art2 (n); */

  free (layers);
  destroy_art2 (n);
}
