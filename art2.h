
#define UNIT_OFF  0.0
#define UNIT_ON   1.0

#define LINEAR      (afn)linear
#define SIGMOID     (afn)sigmoid
#define THRESHOLD   (afn)threshold
#define GAUSSIAN    (afn)gaussian
#define ON_CENTER   (afn)on_center_off_surround

#define RANDOM      0
#define TEST        1

#define DOT_PRODUCT (pfn)dot_product
#define TRANSFER    (pfn)one_to_one

#define COMPLETE    0
#define ONE_TO_ONE  1
#define NORMAL      2
#define VALUE       3

typedef int (*afn) ();          /* type of activation functions     */
typedef void (*pfn) ();         /* type of propagation function     */

class sublayer
{

public:

  float  *w;
  float  *x;
  float  *v;
  float  *u;
  float  *p;
  float  *q;
  float  *r;
 };

class layer          /* the generic layer structure      */
{

public:
   int   units;         /* count of units on layer      */
   int   inputs;        /* count of units feeding this layer    */
   int   processed;     /* index value on some layers       */
   float modifier;      /* modifier for activation function */
   float initval;       /* used to initialize some connections  */
   float *outputs;      /* pointer to array of output values    */
   float **connects;        /* pointer to array of pointers for wts */
   afn   activation;        /* activation function for the layer    */
   pfn   propto;        /* propagation function for layer   */

   #ifdef BPN
     float **lastdelta;     /* used only by the bpn network     */
     float *errors;
     float eta;
     float alpha;
     afn   deriv;
   #endif
};

class art2
{

public:
	art2(float a, float b, float c, float d, float theta, float rho);
	int *define_layers (int layers, ...);
	int *build_art2 (int *sizes);
	void set_art2_parameters (float a, float b, float c, float d, float theta, float rho);
	void connect (layer *inlayer, layer *tolayer, int how, int init);
	void set_parameters (layer *l, pfn p, afn a, float m);
	void set_propagation (layer *l, pfn netx);

private:

	int       layers;		    /* number of layers in the network	*/
	int	    exemplars;		    /* number of training pairs in net	*/
	float     A, B, C, D, theta, rho; /* ART Learning parameters		*/
	float     *errs;		    /* array of error values at output	*/
	iop	    *patterns;		    /* training pairs structure pointer	*/
	class layer     **net;                  /* use the basic network structure	*/
	class sublayer  f1;			    /* F1 sublayer structure		*/
	char      filename[40];	    /* default name for network file	*/
};
