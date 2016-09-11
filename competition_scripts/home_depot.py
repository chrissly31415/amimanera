#!/usr/bin/python
# coding: utf-8

from qsprLib import *
from lasagne_tools import *
from keras_tools import *

import re




sys.path.append('/home/loschen/calc/smuRF/python_wrapper')
import smurf as sf

sys.path.append('/home/loschen/programs/xgboost/wrapper')
import xgboost as xgb

from nlp_features import *
from interact_analysis import *

import matplotlib.pyplot as plt


#from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
from nltk import corpus

#stemmer = SnowballStemmer('english')
stemmer = PorterStemmer() # faster

stop_words = text.ENGLISH_STOP_WORDS.union(corpus.stopwords.words('english'))

def prepareAllFeatures():
    Xtest = pd.read_csv("data/all_test.csv")
    Xtrain = pd.read_csv("data/all_train.csv")
    #r^2>0.98
    correlated = ['sgm_ranks2', 'chris_desc_length', 'kmeans1_sim_title1', 'cluster5_4', 'chris_product_uid', 'carl_fea5', 'len_of_description', 'cluster3_12', 'sgm_ranks4', 'cluster2_16', 'cluster2_13', 'cluster2_12', 'cluster2_11', 'cluster2_10', 'sgm_ranks3', 'cluster4_20', 'cluster4_21', 'cluster4_22', 'cluster5_12', 'kmeans3_sim_description3', 'cluster5_11', 'sgm_ranks1', 'cluster4_9', 'cluster4_8', 'kmeans2_sim_title1', 'cluster4_5', 'cluster4_4', 'cluster4_7', 'cluster4_3', 'cluster4_2', 'cluster2_7', 'cluster2_6', 'cluster2_5', 'cluster2_4', 'cluster2_3', 'cluster2_2', 'cluster2_1', 'cluster3_7', 'kmeans4_sim_title3', 'cluster5_8', 'cluster5_9', 'cluster3_0', 'cluster4_18', 'cluster2_22', 'cluster2_21', 'cluster4_11', 'cluster4_10', 'cluster1_2', 'cluster4_12', 'cluster4_15', 'kmeans2_sim_description1', 'kmeans5_sim_title3', 'kmeans5_sim_description3', 'cluster5_7', 'cluster5_0', 'cluster4_13', 'cluster5_1']
    ytrain = Xtrain['relevance']

    test_id = Xtest['id']
    Xtest.drop(correlated+['id'],axis=1,inplace=True)

    hl = pd.read_csv("./holdout_chrisslz.csv")['id']
    hmask = np.in1d(Xtrain['id'].values, hl.values)
    tmask = np.logical_not(hmask)

    Xval = Xtrain[hmask].copy()
    yval = ytrain[hmask].copy()
    ytrain = ytrain[tmask].copy()
    Xtrain = Xtrain[tmask].copy()

    val_idx = Xval['id']
    Xtrain.drop(correlated+['relevance']+['id'],axis=1,inplace=True)
    Xval.drop(correlated+['relevance']+['id'],axis=1,inplace=True)

    return Xtest, Xtrain, ytrain, Xval, yval,test_id, val_idx

def str_stemmer(s):
    liste = [stemmer.stem(word) for word in s.lower().split()]
    liste = " ".join(liste)
    return(liste)


def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

#with open("spell_check_dict.pkl", "w") as f: pickle.dump(spell_check_dict, f)
with open("spell_check_dict.pkl", "r") as f: spell_check_dict = pickle.load(f)

spell_check_machine = {}
with open('dictionary.txt') as outfile:
    for row in outfile:
        r = row.rstrip()
        x = re.findall("\w+",r)
        if x:
            spell_check_machine[x[0]] = x[1]


with open('spell_check_yang.csv') as outfile:
    outfile.readline()
    for row in outfile:
        k, v = row.rstrip().split(",")
        if k not in spell_check_machine:
            spell_check_machine[k] = v


def train_all():
  Xtest, Xtrain, ytrain, Xval, yval, test_idx, val_idx = prepareAllFeatures()
  #model = Ridge()
  #model = Pipeline([('scaler', StandardScaler()), ('nn',PLSRegression())])
  model  = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=30,learning_rate=8.91E-4,validation_split=0.0,batch_size=256,verbose=1,activation='tanh', layers=[1000,1000], dropout=[0.5,0.0],loss='mse') # best
  model = Pipeline([('scaler', StandardScaler()), ('nn',model)])
  #model = BaggingRegressor(model,n_estimators=3, max_samples=1.0, max_features=1.0, bootstrap=True)
  cv_labels = pd.Series.from_csv('./data/labels_for_cv.csv')
  cv = LabelKFold(cv_labels, n_folds=8)
  #cv = LabelShuffleSplit(cv_labels, n_iter=2)
  scoring_func = make_scorer(root_mean_squared_error, greater_is_better=False)

  print "Evaluation data set..."
  model.fit(Xtrain,ytrain)
  yval_pred = model.predict(Xval)
  print " Eval-score: %5.4f"%(root_mean_squared_error(yval,np.clip(yval_pred,1.0,3.0)))
  df_val = {'id': val_idx, 'pred': yval_pred, 'relevance': yval}
  df_val = pd.DataFrame(df_val)
  df_val.to_csv('bn_val.csv',index=False)

  #Extract hidden layer
  w = model.named_steps['nn'].model.layers[0].get_weights()
  print w[0].shape
  print type(w[0])

  model_best = model.named_steps['nn'].model
  get_feature = theano.function([model_best.layers[0].input],model_best.layers[3].get_output(train=False),allow_input_downcast=True)
  feature = get_feature(Xtrain.values)
  print feature
  print feature.shape

  #model2 = Sequential()
  #model2.add(Dense(632, 1000, weights=w))
  #model2.add(Activation('sigmoid'))
  #act = model2.predict(Xtrain)
  #print act
  #print type(act)

  print "Training the final model (incl. Xval.)"
  Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
  model.fit(Xtrain,ytrain)
  df_test = {'id': test_idx, 'relevance': model.predict(Xtest)}
  df_test = pd.DataFrame(df_test)
  df_test.to_csv('bn_test.csv',index=False)

  #makePredictions(model,Xtest,idx=idx, filename='./submissions/subbaggednn.csv')


#spelling_corrections
typo_pairs = {
 'accesorios': 'accessories',
 'afakro': 'fakro',
 'aire': 'air',
 'airen': 'ariens',
 'airens': 'ariens',
 'airhose': 'air hose',
 'airnailer': 'air nailer',
 'airvents': 'air vents',
 'artifical': 'artificial',
 'akita': 'makita',
 'aluminium': 'aluminum',
 'americna': 'american',
 'aqg': 'awg',
 'aprilaire': 'april air',
 'arcfault': 'arc fault',
 'arien': 'ariens',
 'asher': 'washer',
 'aspectmetal': 'aspect metal',
 'axxes': 'axles',
 'baag': 'bag',
 'bacco': 'basco',
 'bakewarte': 'bakeware',
 'baraar': 'bazaar',
 'barreir': 'barrier',
 'basem': 'base',
 'bathroomvanity': 'bathroom vanity',
 'batte': 'battery',
 'beadez': 'bead ez',
 'behroil': 'behr oil',
 'bidat': 'bidet',
 'bluwood': 'burlwood',
 'bosche': 'bosch',
 'bosh': 'bosch',
 'bostich': 'bostitch',
 'brambury': 'banbury',
 'brayer': 'bayer',
 'brinkman': 'brinkmann',
 'btuair': 'btu air',
 'bulbsu': 'bulb',
 'bungie': 'bungee',
 'bushbutton': 'push button',
 'byass': 'bypass',
 'byefold': 'bi fold',
 'cabients': 'cabinets',
 'cabinetscornor': 'cabinet corner',
 'cabniets': 'cabinets',
 'calk': 'caulk',
 'canapies': 'canopy',
 'canapu': 'canopy',
 'candels': 'candles',
 'carpetshampoo': 'carpet shampoo',
 'carpt': 'carpet',
 'carrera': 'carrara',
 'casing0000': 'case 0000',
 'castel': 'castle',
 'canspot': 'can spot',
 'ceadar': 'cedar',
 'celine': 'ceiling',
 'celing': 'ceiling',
 'cerowir': 'cerrowir',
 'cetolsrd': 'cetol srd',
 'cieling': 'ceiling',
 'claissic': 'classic',
 'clap': 'clamp',
 'cleane': 'cleaner',
 'coatracks': 'coat racks',
 'comercialcarpet': 'commercial carpet',
 'commercail': 'commercial',
 'composit': 'composite',
 'compressro': 'compressor',
 'concretetile': 'concrete tile',
 'condulets': 'conduits',
 'controle': 'control',
 'copoktop': 'cooktop',
 'corne': 'corner',
 'cornershower': 'corner shower',
 'craftsm': 'craftsman',
 'cucbi': 'cuft',
 'cutte': 'cutter',
 'dampiner': 'dampen',
 'deadblow': 'dead blow',
 'deask': 'desk',
 'decarail': 'deckorail',
 'deckpaint': 'deck paint',
 'decostrip': 'deco strip',
 'defence': 'defense',
 'dehimid': 'dehumidify',
 'destiny': 'density',
 'didger': 'digger',
 'dlight': 'light',
 'doly': 'dolly',
 'doo': 'door',
 'dooor': 'door',
 'doorguard': 'door guard',
 'doublesided': 'double sided',
 'dreme': 'dream',
 'drumel': 'dremel',
 'drye': 'dryer',
 'dryed': 'dryer',
 'eathgrow': 'earthgro',
 'edsel': 'edsal',
 'electrica': 'electrical',
 'ensley': 'hensley',
 'entrydoor': 'entry door',
 'entrydoors': 'entry doors',
 'estore': 'restore',
 'evap': 'evaporative',
 'everbuilt': 'everbilt',
 'exteriordoor': 'exterior door',
 'exteriordoors': 'exterior doors',
 'facet': 'faucet',
 'famed': 'framed',
 'fanlight': 'fan light',
 'faul': 'fault',
 'faust': 'faucet',
 'fen': 'fence',
 'ffill': 'fill',
 'fibre': 'fiber',
 'fija': 'fiji',
 'filerts': 'filters',
 'filtrete': 'filter',
 'fireglass': 'fire glass',
 'firepit': 'fire pit',
 'firepits': 'fire pits',
 'firter': 'filter',
 'fj': 'finger joint',
 'floaties': 'float',
 'floorong': 'flooring',
 'floot': 'floor',
 'florescent': 'fluorescent',
 'flotex': 'flotec',
 'flushmount': 'flush mount',
 'foot': 'feet',
 'frenchwood': 'french wood',
 'frontload': 'front load',
 'frose': 'rose',
 'fyrpon': 'fypon',
 'galllon': 'gal',
 'gallondrywall': 'gal drywall',
 'galloon': 'gal',
 'galv': 'galvanized',
 'garde': 'garden',
 'gardena': 'garden',
 'gasquet': 'gasket',
 'gelco': 'gelcoat',
 'gerbera': 'gerber',
 'gfic': 'gfci',
 'gimmbl': 'gimbal',
 'gl': 'gal',
 'goldroyalty': 'gold royalty',
 'greecian': 'grecian',
 'grib': 'grip',
 'gridlee': 'griddle',
 'grils': 'grills',
 'guage': 'gauge',
 'gwtters': 'gutters',
 'hallodoor': 'hollow door',
 'hammerdrill': 'hammer drill',
 'handtools': 'hand tools',
 'harden': 'garden',
 'hardens': 'gardens',
 'hattic': 'attic',
 'heavyduty': 'heavy duty',
 'heightx': 'height x',
 'hexhead': 'hex head',
 'highboy': 'high boy',
 'hindges': 'hinges',
 'homedepot': 'home depot',
 'hookup': 'hook up',
 'hooverwindtunnel': 'hoover wind tunnel',
 'horizontal': 'horizontal',
 'hprse': 'horse',
 'hucky': 'husky',
 'hxh': 'hub x hub',
 'icey': 'ice',
 'iiluminart': 'illuminart',
 'ilumination': 'illumination',
 'inanchor': 'in anchor',
 'inc': 'inch',
 'inchbrasswalltube': 'brass wall tube',
 'inchesside': 'inches side',
 'inchx': 'inch x',
 'inground': 'in ground',
 'inferni': 'inferno',
 'insol': 'insulation',
 'insolation': 'insulation',
 'jacuzzi': 'whirlpool',
 'jb': 'j b',
 'jeldwen': 'jeld wen',
 'jnches': 'inches',
 'joisthangers': 'joist hangers',
 'kalkey': 'kelkay',
 'kegerator': 'kegorator',
 'kelton': 'kelston',
 'keystock': 'key stock',
 'kitchenfaucet': 'kitchen faucet',
 'kitching': 'kitchen',
 'knicken': 'nickel',
 'koehler': 'kohler',
 'kohl': 'kohler',
 'kohlerdrop': 'kohler drop',
 'koolaroo': 'coolaroo',
 'kti': 'kit',
 'kyobi': 'ryobi',
 'lader': 'ladder',
 'laminet': 'laminated',
 'lampt': 'lamp',
 'lanterun': 'lantern',
 'latchbolt': 'latch bolt',
 'lattis': 'lattice',
 'leavers': 'levers',
 'ledbulb': 'led bulb',
 'leuver': 'lever',
 'ligh': 'light',
 'lightbulb': 'light bulb',
 'lightbulbs': 'light bulbs',
 'lightsensor': 'light sensor',
 'ligths': 'lights',
 'lightst': 'lights',
 'litex': 'latex',
 'lithe': 'lithium',
 'lockbox': 'lock box',
 'louvr': 'louver',
 'louvre': 'louver',
 'lover': 'louver',
 'lumder': 'lumber',
 'luminary': 'luminaria',
 'mandare': 'mandara',
 'mashine': 'machine',
 'masterlock': 'master lock',
 'matha': 'martha',
 'mdfrosette': 'mdf rosette',
 'memoir': 'memoirs',
 'microwavekmhc': 'microwave kmhc',
 'mien': 'moen',
 'milwaukie': 'milwaukee',
 'mircowave': 'microwave',
 'miricale': 'miracle',
 'mirro': 'mirror',
 'misters': 'mist',
 'mitre': 'miter',
 'moa': 'mower',
 'mountspacesaver': 'mount space saver',
 'murrial': 'mural',
 'nailgun': 'nail gun',
 'nicket': 'nickel',
 'nickle': 'nickel',
 'nickelshower': 'nickel shower',
 'nickl': 'nickel',
 'oder': 'odor',
 'oerate': 'operated',
 'ofal': 'oval',
 'omen': 'moen',
 'outdoorfurniture': 'outdoor furniture',
 'outlit': 'outlet',
 'owen': 'owens',
 'padspads': 'pads',
 'paintr': 'paint',
 'pak': 'pack',
 'parque': 'parquet',
 'pave': 'paver',
 'paving': 'paver',
 'pecks': 'packs',
 'peelets': 'pellets',
 'pembria': 'pembrey',
 'petscreen': 'pet screen',
 'pfistersaxton': 'pfister saxton',
 'philip': 'philips',
 'phillits': 'philips',
 'pice': 'piece',
 'pipeinless': 'pipe inless',
 'pipeized': 'pipes',
 'pipies': 'pipes',
 'pipy': 'pipe',
 'pk': 'pack',
 'plaers': 'pliers',
 'plastidip': 'plasti dip',
 'plastis': 'plastic',
 'plata': 'plate',
 'plumber': 'plumbers',
 'pnl': 'panel',
 'poste': 'post',
 'posthole': 'post hole',
 'pound': 'lb',
 'powrcoat': 'procoat',
 'preiumer': 'primer',
 'prelit': 'pre lit',
 'pvs': 'pvc',
 'qquart': 'qt',
 'ragne': 'range',
 'raido': 'radio',
 'rainbarrel': 'rain barrel',
 'rainspout': 'rain spout',
 'rak': 'rack',
 'ranshower': 'rain shower',
 'rebarbender': 'rebar bender',
 'reffridge': 'refrigerator',
 'refrig': 'refrigerator',
 'refrigeratorators': 'refrigerators',
 'rehrig': 'refrigerator',
 'reliabilt': 'reliable',
 'replament': 'replacement',
 'replaclacemt': 'replacement',
 'rheum': 'rheem',
 'roby': 'ryobi',
 'rockwool': 'rock wool',
 'role': 'roll',
 'roles': 'rolls',
 'rools': 'tools',
 'rotomartillo': 'rotary',
 'rototiller': 'roto tiller',
 'rudolf': 'rudolph',
 'ruf': 'rug',
 'safavids': 'safavieh',
 'sawal': 'sawzal',
 'sawall': 'sawzal',
 'saww': 'saw',
 'sched': 'schedule',
 'sedgehammer': 'sledge hammer',
 'seedeater': 'trimmer',
 'selfstick': 'self stick',
 'selves': 'shelves',
 'semirigid': 'semi rigid',
 'seperate': 'separate',
 'seriestruease': 'series truease',
 'shaanchor': 'shape anchor',
 'sharer': 'shaker',
 'sheetin': 'sheet',
 'shefl': 'shelf',
 'shelfa': 'shelf',
 'shelflike': 'shelf',
 'shels': 'shelves',
 'shepard': 'shepherd',
 'shepards': 'shepherd',
 'sheves': 'shelves',
 'shoplight': 'shop light',
 'showerheards': 'shower heads',
 'showerstall': 'shower stall',
 'showerz': 'shower',
 'siclica': 'silica',
 'sikalatex': 'sika latex',
 'sillcick': 'sillcock',
 'silocoln': 'silicon',
 'sin': 'sink',
 'singl': 'single',
 'sistem': 'system',
 'sofn': 'soften',
 'solidconcrete': 'solid concrete',
 'sower': 'sewer',
 'spicket': 'spigot',
 'spraypaint': 'spray paint',
 'steamwasher': 'steam washer',
 'steal': 'steel',
 'steamfresh': 'steam fresh',
 'steele': 'steel',
 'sterlite': 'sterilite',
 'stnless': 'stainless',
 'stnls': 'stainless steel',
 'stiprs': 'stripes',
 'stopwa': 'stop',
 'storeges': 'storage',
 'strippping': 'stripping',
 'stronglus': 'strong',
 'subpumps': 'sump pumps',
 'sylvester': 'sylvestre',
 'syston': 'system',
 'tapecase': 'tape case',
 'tarpaulin': 'tarp',
 'tbl': 'table',
 'thck': 'thick',
 'thinset': 'thin set',
 'tighrner': 'tighten',
 'timmers': 'timers',
 'tomostat': 'thermostat',
 'toplass': 'top',
 'toprail': 'top rail',
 'topsealer': 'top sealer',
 'tordon': 'gordon',
 'toulone': 'toulon',
 'trashcan': 'trash can',
 'trcat': 'tractor',
 'treillis': 'trellis',
 'tresers': 'dressers',
 'treshold': 'threshold',
 'trimec': 'trimaco',
 'trimer': 'trimmer',
 'tu': 'tub',
 'turbi': 'turbine',
 'unsulation': 'insulation',
 'vaccum': 'vacuum',
 'vacum': 'vacuum',
 'vacume': 'vacuum',
 'valleriana': 'valeriano',
 'vegetale': 'vegetable',
 'venner': 'veneer',
 'ventless': 'vent free',
 'vigaro': 'vigoro',
 'vinal': 'vinyl',
 'vintemp': 'vinotemp',
 'waddles': 'saddles',
 'waher': 'washer',
 'wallx': 'wall x',
 'walmound': 'wall mount',
 'washer000': 'washer 000',
 'wastel': 'waste',
 'waterheater': 'water heater',
 'weatherstriping': 'weather strip',
 'weedbgon': 'weed b gon',
 'wera': 'wear',
 'whearehoues': 'warehouse',
 'wiga': 'wigan',
 'windos': 'windows',
 'windowtioners': 'window',
 'wiremesh': 'wire mesh',
 'wirless': 'wireless',
 'withx': 'with x',
 'woo': 'wood',
 'woodflooring': 'wood flooring',
 'woodstove': 'wood stove',
 'woodstoves': 'wood stoves',
 'worklight': 'work light',
 'worklite': 'work light',
 'xmas': 'christmas',
 'yardguard': 'yardgard',
 'yr': 'year',
 'zeroturn': 'zero turn',
 'zwave': 'z wave'
}

for k, v in typo_pairs.iteritems():
    if k not in spell_check_machine:
        spell_check_machine[k] = v

### another replacement dict used independently
another_replacement_dict={"undercabinet": "under cabinet",
"snowerblower": "snower blower",
"mountreading": "mount reading",
"zeroturn": "zero turn",
"stemcartridge": "stem cartridge",
"greecianmarble": "greecian marble",
"outdoorfurniture": "outdoor furniture",
"outdoorlounge": "outdoor lounge",
"heaterconditioner": "heater conditioner",
"heater/conditioner": "heater conditioner",
"conditioner/heater": "conditioner heater",
"airconditioner": "air conditioner",
"snowbl": "snow bl",
"plexigla": "plexi gla",
"whirlpoolga": "whirlpool ga",
"whirlpoolstainless": "whirlpool stainless",
"sedgehamm": "sledge hamm",
"childproof": "child proof",
"flatbraces": "flat braces",
"zmax": "z max",
"gal vanized": "galvanized",
"battery powere weedeater": "battery power weed eater",
"shark bite": "sharkbite",
"rigid saw": "ridgid saw",
"black decke": "black and decker",
"exteriorpaint": "exterior paint",
"fuelpellets": "fuel pellet",
"cabinetwithouttops": "cabinet without tops",
"castiron": "cast iron",
"pfistersaxton": "pfister saxton ",
"splitbolt": "split bolt",
"soundfroofing": "sound froofing",
"cornershower": "corner shower",
"stronglus": "strong lus",
"shopvac": "shop vac",
"shoplight": "shop light",
"airconditioner": "air conditioner",
"whirlpoolga": "whirlpool ga",
"whirlpoolstainless": "whirlpool stainless",
"snowblower": "snow blower",
"plexigla": "plexi gla",
"trashcan": "trash can",
"mountspacesaver": "mount space saver",
"undercounter": "under counter",
"stairtreads": "stair tread",
"techni soil": "technisoil",
"in sulated": "insulated",
"closet maid": "closetmaid",
"we mo": "wemo",
"weather tech": "weathertech",
"weather vane": "weathervane",
"versa tube": "versatube",
"versa bond": "versabond",
"in termatic": "intermatic",
"therma cell": "thermacell",
"tuff screen": "tuffscreen",
"sani flo": "saniflo",
"timber lok": "timberlok",
"thresh hold": "threshold",
"yardguard": "yardgard",
"incyh": "in.",
"diswasher": "dishwasher",
"closetmade": "closetmaid",
"repir": "repair",
"handycap": "handicap",
"toliet": "toilet",
"conditionar": "conditioner",
"aircondition": "air conditioner",
"aircondiioner": "air conditioner",
"comercialcarpet": "commercial carpet",
"commercail": "commercial",
"inyl": "vinyl",
"vinal": "vinyl",
"vynal": "vinyl",
"vynik": "vinyl",
"skill": "skil",
"whirpool": "whirlpool",
"glaciar": "glacier",
"glacie": "glacier",
"rheum": "rheem",
"one+": "1",
"toll": "tool",
"ceadar": "cedar",
"shelv": "shelf",
"toillet": "toilet",
"toiet": "toilet",
"toilest": "toilet",
"toitet": "toilet",
"ktoilet": "toilet",
"tiolet": "toilet",
"tolet": "toilet",
"eater": "heater",
"robi": "ryobi",
"robyi": "ryobi",
"roybi": "ryobi",
"rayobi": "ryobi",
"riobi": "ryobi",
"screww": "screw",
"stailess": "stainless",
"dor": "door",
"vaccuum": "vacuum",
"vacum": "vacuum",
"vaccum": "vacuum",
"vinal": "vinyl",
"vynal": "vinyl",
"vinli": "vinyl",
"viyl": "vinyl",
"vynil": "vinyl",
"vlave": "valve",
"vlve": "valve",
"walll": "wall",
"steal": "steel",
"stell": "steel",
"pcv": "pvc",
"blub": "bulb",
"ligt": "light",
"bateri": "battery",
"kolher": "kohler",
"fame": "frame",
"have": "haven",
"acccessori": "accessory",
"accecori": "accessory",
"accesnt": "accessory",
"accesor": "accessory",
"accesori": "accessory",
"accesorio": "accessory",
"accessori": "accessory",
"repac": "replacement",
"repalc": "replacement",
"repar": "repair",
"repir": "repair",
"replacemet": "replacement",
"replacemetn": "replacement",
"replacemtn": "replacement",
"replaclacemt": "replacement",
"replament": "replacement",
"toliet": "toilet",
"skill": "skil",
"whirpool": "whirlpool",
"stailess": "stainless",
"stainlss": "stainless",
"stainstess": "stainless",
"jigsaww": "jig saw",
"woodwen": "wood",
"pywood": "plywood",
"woodebn": "wood",
"repellant": "repellent",
"concret": "concrete",
"windos": "window",
"wndows": "window",
"wndow": "window",
"winow": "window",
"caamera": "camera",
"sitch": "switch",
"doort": "door",
"coller": "cooler",
"flasheing": "flashing",
"wiga": "wigan",
"bathroon": "bath room",
"sinl": "sink",
"melimine": "melamine",
"inyrtior": "interior",
"tilw": "tile",
"wheelbarow": "wheelbarrow",
"pedistal": "pedestal",
"submerciable": "submercible",
"weldn": "weld",
"contaner": "container",
"webmo": "wemo",
"genis": "genesis",
"waxhers": "washer",
"softners": "softener",
"sofn": "softener",
"connecter": "connector",
"heather": "heater",
"heatere": "heater",
"electic": "electric",
"quarteround": "quarter round",
"bprder": "border",
"pannels": "panel",
"framelessmirror": "frameless mirror",
"paneling": "panel",
"controle": "control",
"flurescent": "fluorescent",
"flourescent": "fluorescent",
"molding": "moulding",
"lattiace": "lattice",
"barackets": "bracket",
"vintemp": "vinotemp",
"vetical": "vertical",
"verticle": "vertical",
"vesel": "vessel",
"versatiube": "versatube",
"versabon": "versabond",
"dampr": "damper",
"vegtable": "vegetable",
"plannter": "planter",
"fictures": "fixture",
"mirros": "mirror",
"topped": "top",
"preventor": "breaker",
"traiter": "trailer",
"ureka": "eureka",
"uplihght": "uplight",
"upholstry": "upholstery",
"untique": "antique",
"unsulation": "insulation",
"unfinushed": "unfinished",
"verathane": "varathane",
"ventenatural": "vent natural",
"shoer": "shower",
"floorong": "flooring",
"tsnkless": "tankless",
"tresers": "dresers",
"treate": "treated",
"transparant": "transparent",
"transormations": "transformation",
"mast5er": "master",
"anity": "vanity",
"tomostat": "thermostat",
"thromastate": "thermostat",
"kphler": "kohler",
"tji": "tpi",
"cuter": "cutter",
"medalions": "medallion",
"tourches": "torch",
"tighrner": "tightener",
"thewall": "the wall",
"thru": "through",
"wayy": "way",
"temping": "tamping",
"outsde": "outdoor",
"bulbsu": "bulb",
"ligh": "light",
"swivrl": "swivel",
"switchplate": "switch plate",
"swiss+tech": "swiss tech",
"sweenys": "sweeney",
"susbenders": "suspender",
"cucbi": "cu",
"gaqs": "gas",
"structered": "structured",
"knops": "knob",
"adopter": "adapter",
"patr": "part",
"storeage": "storage",
"venner": "veneer",
"veneerstone": "veneer stone",
"stm": "stem",
"steqamers": "steamer",
"latter": "ladder",
"steele": "steel",
"builco": "bilco",
"panals": "panel",
"grasa": "grass",
"unners": "runner",
"maogani": "maogany",
"sinl": "sink",
"grat": "grate",
"showerheards": "shower head",
"spunge": "sponge",
"conroller": "controller",
"cleanerm": "cleaner",
"preiumer": "primer",
"fertillzer": "fertilzer",
"spectrazide": "spectracide",
"spaonges": "sponge",
"stoage": "storage",
"sower": "shower",
"solor": "solar",
"sodering": "solder",
"powerd": "powered",
"lmapy": "lamp",
"naturlas": "natural",
"sodpstone": "soapstone",
"punp": "pump",
"blowerr": "blower",
"medicn": "medicine",
"slidein": "slide",
"sjhelf": "shelf",
"oard": "board",
"singel": "single",
"paintr": "paint",
"silocoln": "silicon",
"poinsetia": "poinsettia",
"sammples": "sample",
"sidelits": "sidelight",
"nitch": "niche",
"pendent": "pendant",
"shopac": "shop vac",
"shoipping": "shopping",
"shelfa": "shelf",
"cabi": "cabinet",
"nails18": "nail",
"dewaqlt": "dewalt",
"barreir": "barrier",
"ilumination": "illumination",
"mortice": "mortise",
"lumes": "lumen",
"blakck": "black",
"exterieur": "exterior",
"expsnsion": "expansion",
"air condit$": "air conditioner",
"double pole type chf breaker": "double pole type ch breaker",
"mast 5 er": "master",
"toilet rak": "toilet rack",
"govenore": "governor",
"in wide": "in white",
"shepard hook": "shepherd hook",
"frost fee": "frost free",
"kitchen aide": "kitchen aid",
"saww horse": "saw horse",
"weather striping": "weatherstripper",
"'girls": "girl",
"girl's": "girl",
"girls'": "girl",
"girls": "girl",
"girlz": "girl",
"boy's": "boy",
"boys'": "boy",
"boys": "boy",
"men's": "man",
"mens'": "man",
"mens": "mam",
"men": "man",
"women's": "woman",
"womens'": "woman",
"womens": "woman",
"women": "woman",
"kid's": "kid",
"kids'": "kid",
"kids": "kid",
"children's": "kid",
"childrens'": "kid",
"childrens": "kid",
"children": "kid",
"child": "kid",
"bras": "bra",
"bicycles": "bike",
"bicycle": "bike",
"bikes": "bike",
"refridgerators": "fridge",
"refrigerator": "fridge",
"refrigirator": "fridge",
"freezer": "fridge",
"memories": "memory",
"fragance": "perfume",
"fragrance": "perfume",
"cologne": "perfume",
"anime": "animal",
"assassinss": "assassin",
"assassin's": "assassin",
"assassins": "assassin",
"bedspreads": "bedspread",
"shoppe": "shop",
"extenal": "external",
"knives": "knife",
"kitty's": "kitty",
"levi's": "levi",
"squared": "square",
"rachel": "rachael",
"rechargable": "rechargeable",
"batteries": "battery",
"seiko's": "seiko",
"ounce": "oz"
}

for k, v in another_replacement_dict.iteritems():
    if k not in spell_check_machine:
        spell_check_machine[k] = v

print "Spell_check_dict:",len(spell_check_machine.keys())

with open("query_map.pkl", "r") as f: query_map = pickle.load(f)#key query value corrected value

def spell_checking(s):
    if s.lower() in spell_check_dict.keys():
        s = spell_check_dict[s]
    return s

def new_spell_checking(s):
    if s in spell_check_machine.keys():
        s = spell_check_machine[s]
    return s

def query_correct(s):
    if s.lower() in query_map.keys():
        s = query_map[s]
    return s


def cleanse_text(s):
    """
    https://www.kaggle.com/vabatista/home-depot-product-search-relevance/test-script-1/code
    """

    if isinstance(s, str) or isinstance(s, unicode):
        #print "before:",s
        s = s.lower()
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) ##'desgruda' palavras que estão juntas

        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)

        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)

        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)

        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")

        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)

        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)

        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)

        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)

        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)

        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)

        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)

        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)

        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)

        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        #print "after:",s
        #raw_input()
        return s.lower()
    else:
        return "null"


def prepareDataset(quickload=False, seed=123, nsamples=-1, holdout=False, keepFeatures=None, dropFeatures = None,dummy_encoding=None,labelEncode=None, oneHotenc=None,removeRare_freq=None, logtransform = None, stemmData=False,  createCommonWords=False, useAttributes= False, doTFIDF= None, computeSim = None, n_components = 20, cleanData = False, merge_product_infos = None,computeAddFeates=False, computeAddFeates_new = False, word2vecFeates= False, word2vecFeates_new = False, spellchecker = False, removeCorr=False, query_correction=False, createVerticalFeatures=False, use_new_spellchecker=False,createVerticalFeatures_new=False):
    print("Preparing data set!")
    np.random.seed(seed)

    if isinstance(quickload,str):
        store = pd.HDFStore(quickload)
        print store
        Xtest = store['Xtest']
        Xtrain = store['Xtrain']
        ytrain = store['ytrain']
        Xval = store['Xval']
        yval = store['yval']
        test_id = store['test_id']

        return Xtest, Xtrain, ytrain.values, test_id, None, Xval, yval.values

    store = pd.HDFStore('./data/store.h5')
    #print store

    Xtrain = pd.read_csv('./data/train.csv', encoding="ISO-8859-1")


    print Xtrain.describe(include='all')
    print "Xtrain.shape:",Xtrain.shape

    Xtest = pd.read_csv('./data/test.csv', encoding="ISO-8859-1")
    print "Xtest.shape:",Xtest.shape


    print "Xtrain - ISNULL:",Xtrain.isnull().any(axis=0)
    print "Xtest - ISNULL:",Xtest.isnull().any(axis=0)

    if nsamples != -1:
        if isinstance(nsamples, str) and 'shuffle' in nsamples:
            print "Shuffle train data..."
            rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index), replace=False)
        else:
            rows = np.random.choice(len(Xtrain.index), size=nsamples, replace=False)

        print "unique rows: %6.2f" % (float(np.unique(rows).shape[0]) / float(rows.shape[0]))
        Xtrain = Xtrain.iloc[rows, :]

    #Xtrain.groupby([Xtrain.Date.dt.year,Xtrain.Date.dt.month])['Sales'].mean().plot(kind="bar")
    #Xtest.groupby([Xtest.Date.dt.year,Xtest.Date.dt.month])['Store'].mean().plot(kind="bar")

    #rearrange
    ytrain = Xtrain['relevance']
    train_id = Xtrain['id']
    Xtrain.drop(['relevance','id'],axis=1,inplace=True)
    test_id = Xtest['id']
    store['test_id'] = test_id
    Xtest.drop(['id'],axis=1,inplace=True)

    #check
    print Xtrain.shape
    print Xtest.shape

    print "Analyzing search terms"
    uniq_train = Xtrain['search_term'].unique()
    uniq_test = Xtest['search_term'].unique()
    compareList(uniq_train, uniq_test, verbose=True)

    uniq_train = Xtrain['product_uid'].unique()
    uniq_test = Xtest['product_uid'].unique()
    compareList(uniq_train, uniq_test, verbose=True)
    #raw_input()

    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)

    df_product_desc = pd.read_csv('./data/product_descriptions.csv',encoding="ISO-8859-1")

    Xall = pd.merge(Xall, df_product_desc, how='left', on='product_uid')


    if useAttributes is not None:
        Xattr = pd.read_csv('./data/attributes.csv',encoding="ISO-8859-1")
        print Xattr.head(100)
        print Xattr['name'].value_counts()
        #raw_input()
        # merge bullet02 bullet03 bullet04 bullet01 product width, bullet04,
        # color is multiple fields
        # https://www.kaggle.com/briantc/home-depot-product-search-relevance/homedepot-first-dataexploreation-k/notebook
        #attributes = ['MFG Brand Name','Material']


        for att in useAttributes:
            # merge attr
            Xbrand = Xattr[Xattr.name == att][["product_uid", "value"]].rename(columns={"value": att})
            Xbrand.drop_duplicates(subset=['product_uid'],inplace=True)
            #Xbrand = Xbrand.astype(str)
            Xall = pd.merge(Xall, Xbrand, how='left', on='product_uid')

        #Xattr.rename(columns={'MFG Brand Name':'brand'},inplace=True)
        #merge material
        #Xmat = Xattr[Xattr.name == "Material"][["product_uid", "value"]].rename(columns={"value": "material"})
        #Xmat.drop_duplicates(subset=['product_uid'],inplace=True)
        #Xall = pd.merge(Xall, Xmat, how='left', on='product_uid')
        Xall.fillna("NA", inplace=True)


    if use_new_spellchecker:
        print "New spell checker for search term..."
        Xall['search_term'] = Xall['search_term'].map(lambda x:new_spell_checking(x))

    if query_correction:
        print "Query correction..."
        #query_map = build_query_correction_map(Xall[len(Xtest.index):],Xall[len(Xtest.index):])
        #with open("query_map.pkl", "w") as f: pickle.dump(query_map, f)
        Xall["search_term"] = Xall["search_term"].map(lambda x:query_correct(x))

    if spellchecker:
        #https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos/discussion
        print "Google Spellchecker for search term..."
        Xall['search_term'] = Xall['search_term'].map(lambda x:spell_checking(x))


    if cleanData is not None:

        if isinstance(cleanData,str):
            print "Loading cleaned words..."
            Xall['search_term'] = store['search_term']
            Xall['product_title'] = store['product_title']
            Xall['product_description'] = store['product_description']
        else:
            print "Cleaning data..."
            Xall['search_term'] = Xall['search_term'].map(lambda x:cleanse_text(x))
            print "Cleaning search_term ok"
            Xall['product_title'] = Xall['product_title'].map(lambda x:cleanse_text(x))
            print "Cleaning product_title ok"
            Xall['product_description'] = Xall['product_description'].map(lambda x:cleanse_text(x))
            print "Cleaning product_description ok"
            store['search_term'] = Xall['search_term']
            store['product_title'] = Xall['product_title']
            store['product_description'] = Xall['product_description']
            print "Saving data ok"

    if  stemmData is not None:

        if isinstance(stemmData,str):
            print "Loading stemmed words..."
            Xall['search_term'] = store['search_term']
            Xall['product_title'] = store['product_title']
            Xall['product_description'] = store['product_description']

        else:
            print "Stemming words..."

            Xall['search_term'] = Xall['search_term'].map(lambda x:str_stemmer(x))
            Xall['product_title'] = Xall['product_title'].map(lambda x:str_stemmer(x))
            Xall['product_description'] = Xall['product_description'].map(lambda x:str_stemmer(x))
            store['search_term'] = Xall['search_term']
            store['product_title'] = Xall['product_title']
            store['product_description'] = Xall['product_description']

    if merge_product_infos or isinstance(merge_product_infos,list):

        if isinstance(merge_product_infos,list):
            print "Merge Cols:",merge_product_infos
            Xall['product_info'] = ""
            for col in merge_product_infos:
                Xall['product_info'] += Xall['product_info']+"\t"+Xall[col]

        else:
            print "Merging title and description"
            Xall['product_info'] = Xall['product_title']+"\t"+Xall['product_description']

    if createVerticalFeatures:
        print "Creating vertical features..."
        st = Xall['search_term'].unique()
        print st.shape
        #Xall['max_quantity'] = 0
        #Xall['mean_quantity'] = 0
        Xall['n_positions'] = 0
        for i, uv in enumerate(st):
            bool_idx = Xall['search_term'] == uv
            #Xall.loc[bool_idx, ['max_quantity']] = Xall.loc[bool_idx, ['quantity']].max(axis=0).values
            #Xall.loc[bool_idx, ['mean_quantity']] = Xall.loc[bool_idx, ['quantity']].median(axis=0).values
            Xall.loc[bool_idx, ['n_positions']] = bool_idx.sum()
            #Xall.loc[bool_idx, ['max_quantity']]
            #print Xall.loc[bool_idx, ['n_positions']]
            #Xall.loc[bool_idx, ['mean_quantity']]
            # raw_input()
            if i % 5000 == 0:
                print "iteration %d/%d" % (i, len(st))

    if createVerticalFeatures_new:
        print "Another set of vertical features..."
        m1 = pd.rolling_mean(ytrain, window=3000, min_periods=1)
        m1.plot()
        sub = pd.read_csv('./submissions/sub150416a.csv')
        m2 = pd.rolling_mean(sub.relevance, window=3000, min_periods=1)
        m = pd.concat([m2,m1],axis=0)
        print m
        print m.shape
        Xall['rolling_mean'] = m.values


    if word2vecFeates:
        print "Add w2vec features"
        if isinstance(word2vecFeates,str):
            print "Loading features data..."
            Xfeat = store['Xword2vec']
        else:
            Xfeat = genWord2VecFeatures(Xall,verbose=False)
            store['Xword2vec'] = Xfeat

        print Xfeat.describe()
        Xall = pd.concat([Xall,Xfeat], axis=1)

    if word2vecFeates_new:
        print "Add w2vec features 2"
        if isinstance(word2vecFeates_new,str):
            print "Loading features data..."
            Xfeat = store['Xword2vec_new']
        else:
            Xfeat = genWord2VecFeatures_new(Xall,verbose=False)
            store['Xword2vec_new'] = Xfeat

        print Xfeat.describe()
        Xall = pd.concat([Xall,Xfeat], axis=1)

    if computeAddFeates:
        print "Add features"
        if isinstance(computeAddFeates,str):
            print "Loading features data..."
            Xfeat = store['Xfeat']
        else:
            Xfeat = additionalFeatures(Xall,verbose=False,dropList=['bestmatch'])
            store['Xfeat'] = Xfeat

        print Xfeat.describe()
        Xall = pd.concat([Xall,Xfeat], axis=1)


    if computeAddFeates_new:
        print "Add features new"
        if isinstance(computeAddFeates_new,str):
            print "Loading features data..."
            Xfeat = store['Xfeat2']
        else:
            Xfeat = additionalFeatures_new(Xall,verbose=False,dropList=['bestmatch'])
            store['Xfeat2'] = Xfeat

        print Xfeat.describe()
        Xall = pd.concat([Xall,Xfeat], axis=1)



    Xsim = None
    if computeSim is not None:
        reducer = None
        if 'reducer' in computeSim:
            reducer=computeSim['reducer']
        Xsim = computeSimilarityFeatures(Xall,columns=computeSim['columns'],verbose=False,useOnlyTrain=False,stop_words=stop_words,doSVD=computeSim['doSVD'],vectorizer=computeSim['vectorizer'],reducer=reducer)
        print Xsim.describe()
        Xall = pd.concat([Xall,Xsim], axis=1)

    if doTFIDF is not None:
        print "Doing TFIDF:",doTFIDF
        vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
        if isinstance(doTFIDF,dict):
            print doTFIDF.keys()
            vectorizer = doTFIDF['vectorizer']
            doTFIDF = doTFIDF['columns']
        for i,(n_comp,col) in enumerate(zip(n_components,doTFIDF)):
            print Xall.shape
            print "Xall - ISNULL:",Xall.isnull().any(axis=0)
            print "Vectorizing: "+col+" n_components:"+str(n_comp)
            vectorizer.fit(Xall[col])
            Xs_all_new = vectorizer.transform(Xall[col])
            reducer=TruncatedSVD(n_components=n_comp, algorithm='randomized', n_iter=5, tol=0.0)
            Xs_all_new=reducer.fit_transform(Xs_all_new)
            print "Variance explained:",np.sum(reducer.explained_variance_ratio_)
            Xs_all_new = pd.DataFrame(Xs_all_new)
            Xs_all_new.columns = [ col+"_svd_"+str(x) for x in Xs_all_new.columns ]
            #store['Xs_all_new'] = Xs_all_new

            Xall = pd.concat([Xall, Xs_all_new], axis=1)

        print "Shape Xs_all after SVD:",Xall.shape



    if createCommonWords:
        """
        See https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest/code
        also: https://www.kaggle.com/the1owl/home-depot-product-search-relevance/rf-mean-squared-error/code
        """
        print "Create common words..."
        Xall['len_of_query'] = Xall['search_term'].map(lambda x:len(x.split())).astype(np.int64)
        Xall['len_of_title'] = Xall['product_title'].map(lambda x:len(x.split())).astype(np.int64)
        Xall['len_of_description'] = Xall['product_description'].map(lambda x:len(x.split())).astype(np.int64)

        Xall['product_info'] = Xall['search_term']+"\t"+Xall['product_title']+"\t"+Xall['product_description']

        Xall['word_in_title'] = Xall['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
        Xall['word_in_description'] = Xall['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

        print Xall.head(10)


    if dummy_encoding is not None:
        print "Dummy encoding,skip label encoding"
        Xall = pd.get_dummies(Xall,columns=dummy_encoding)

    if labelEncode is not None:
        print "Label encode"
        for col in labelEncode:
            lbl = preprocessing.LabelEncoder()
            Xall[col] = lbl.fit_transform(Xall[col].values)
            vals = Xall[col].unique()
            print "Col: %s Vals %r:"%(col,vals)
            #print "Orig:",list(lbl.inverse_transform(Xall[col].unique()))

    if removeRare_freq is not None:
        print "Remove rare features based on frequency..."
        for col in oneHotenc:
            ser = Xall[col]
            counts = ser.value_counts().keys()
            idx = ser.value_counts() > removeRare_freq
            threshold = idx.astype(int).sum()
            print "%s has %d different values before, min freq: %d - threshold %d" % (
                col, len(counts), removeRare_freq, threshold)
            if len(counts) > threshold:
                ser[~ser.isin(counts[:threshold])] = 9999
            if len(counts) <= 1:
                print("Dropping Column %s with %d values" % (col, len(counts)))
                Xall = Xall.drop(col, axis=1)
            else:
                Xall[col] = ser.astype('category')
            print ser.value_counts()
            counts = ser.value_counts().keys()
            print "%s has %d different values after" % (col, len(counts))


    if oneHotenc is not None:
        print "1-0 Encoding categoricals...", oneHotenc
        for col in oneHotenc:
            #print "Unique values for col:", col, " -", np.unique(Xall[col].values)
            encoder = OneHotEncoder()
            X_onehot = pd.DataFrame(encoder.fit_transform(Xall[[col]].values).todense())
            X_onehot.columns = [col + "_hc_" + str(column) for column in X_onehot.columns]
            print "One-hot-encoding of %r...new shape: %r" % (col, X_onehot.shape)
            Xall.drop([col], axis=1, inplace=True)
            Xall = pd.concat([Xall, X_onehot], axis=1)
            print "One-hot-encoding final shape:", Xall.shape
            # raw_input()

    if logtransform is not None:
        print "log Transform"
        for col in logtransform:
            if col in Xall.columns:
                if Xall[col].min()>-0.99:
                    Xall[col] = Xall[col].map(np.log1p+1.0)


    if keepFeatures is not None:
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        for col in dropcols:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)
        Xall.sort(columns=keepFeatures,inplace=True)

    if dropFeatures is not None:
        for col in dropFeatures:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    if removeCorr:
        Xall = removeCorrelations(Xall, threshhold=0.99)

    #Xall = Xall.astype(np.float32)
    print "Columns used",list(Xall.columns)


    #split data
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    Xval = None
    yval = pd.Series()

    if isinstance(holdout,str) or holdout:
        print "Split holdout..."
        if isinstance(holdout,str):
            #hl = pd.Series.from_csv(holdout,index_col=None)
            hl = pd.read_csv(holdout)['id']
            hmask = np.in1d(train_id.values, hl.values)
            tmask = np.logical_not(hmask)

            Xval = Xtrain[hmask]
            yval = ytrain[hmask]

            ytrain = ytrain[tmask]
            Xtrain = Xtrain[tmask]

        else:
            #Xtrain['search_term'].to_csv('./data/labels_for_holdout.csv')
            holdout_labels = pd.Series.from_csv('./data/labels_for_holdout.csv')
            #Xtrain, Xval, ytrain, yval = label_train_test_plit(Xtrain,ytrain,labels=Xtrain['search_term'],test_size=0.3, random_state=666)

            Xtrain, Xval, ytrain, yval = label_train_test_plit(Xtrain,ytrain,labels=holdout_labels,test_size=0.3, random_state=5711)

            print ytrain
            print ytrain.shape

            #Xtrain['search_term'].to_csv('./data/labels_for_cv.csv')

        print Xtrain.head()
        print Xval.head()
        print "Shape Xtrain:",Xtrain.shape
        print "Shape Xval  :",Xval.shape
        print ytrain
        print ytrain.shape
        #raw_input()

    print "Training data:",Xtrain.info()

    df_list = [Xtest,Xtrain,ytrain,Xval,yval,test_id,Xtrain[['search_term']]]
    name_list = ['Xtest','Xtrain','ytrain','Xval','yval','test_id','cv_labels']
    for label,ldf in zip(name_list,df_list):
        if ldf is not None:
            try:
                print "Store:",label
                ldf = ldf.reindex(copy = False)
                store.put(label, ldf, format='table', data_columns=True)
            except:
                print "Could not save to pytables:"
                print ldf.apply(lambda x: pd.lib.infer_dtype(x.values))

    store.close()

    return Xtest, Xtrain, ytrain.values, test_id, None, Xval, yval.values


def post_process(yval_pred,yval,sigma = 0.05,k = 0.5):
    attractors = np.unique(yval)
    #attractors = [1.0,1.33,1.67,2.0,2.33,2.67,3.0]
    #attractors = [1.0,3.0]
    #print attractors
    yval_pred_orig = yval_pred.copy()
    for mu in attractors:
        damp = np.exp(-(yval_pred-mu)**2/(2*sigma**2))
        shift =  k*(mu-yval_pred)
        yval_pred = yval_pred + shift *damp

    plt.hist(yval,bins=50)
    #plt.hist(yval_pred_orig,bins=50)
    plt.hist(yval_pred,bins=50)
    plt.show()
    return yval_pred

def mergeWithXval(Xtrain,Xval,ytrain,yval):
    Xtrain =  pd.concat([Xtrain, Xval], ignore_index=True)
    ytrain = np.hstack((ytrain,yval))
    return Xtrain,ytrain


def makePredictions(model=None,Xtest=None,idx=None,filename='submission.csv'):
    print "Saving submission: ", filename
    if model is not None:
        preds = model.predict(Xtest)
    else:
        preds = Xtest
    if idx is None:
        idx = np.arange(Xtest.shape[0])+1

    result = pd.DataFrame({"id": idx, 'relevance': preds})

    result['relevance'] = result['relevance'].clip(1.0,3.0)
    result.to_csv(filename, index=False)


if __name__ == "__main__":
    """
    MAIN PART
    """
    pd.options.display.mpl_style = 'default'
    plt.interactive(False)
    #TODO
    # one hot encoding search term
    # cleanse!!!
    # look at https://github.com/ChenglongChen/Kaggle_CrowdFlower/blob/master/Doc/Kaggle_CrowdFlower_ChenglongChen.pdf
    # truncate predictions
    # use product id for cross_validation
    # coorect typos gsub("([a-z])([A-Z])([a-z])", "\\1 \\2\\3", product_description)
    # vertical features for search query / product_id
    # searchterm as label!!! groupby
    # merge title information
    # http://norvig.com/spell-correct.html
    # spelling correction ->pyenchant

    t0 = time()

    print "numpy:", np.__version__
    print "pandas:", pd.__version__
    print "scipy:", sp.__version__

    #TODO
    # use attributes!!
    # cleanse text
    # word2vec

    all_feats = [u'product_uid', u'search_term', 'query_length', 'title_length', 'query_title_ratio', 'desc_length', 'query_desc_ratio', 'difflibratio', 'averagematch', 'S_query', 'S_title', 'last_sim', 'first_sim', 'checksynonyma', 'cosine', 'cityblock', 'hamming', 'euclidean', 'search_term_svd_0', 'search_term_svd_1', 'search_term_svd_2', 'search_term_svd_3', 'search_term_svd_4', 'search_term_svd_5', 'search_term_svd_6', 'search_term_svd_7', 'search_term_svd_8', 'search_term_svd_9', 'search_term_svd_10', 'search_term_svd_11', 'search_term_svd_12', 'search_term_svd_13', 'search_term_svd_14', 'search_term_svd_15', 'search_term_svd_16', 'search_term_svd_17', 'search_term_svd_18', 'search_term_svd_19', 'search_term_svd_20', 'search_term_svd_21', 'search_term_svd_22', 'search_term_svd_23', 'search_term_svd_24', 'search_term_svd_25', 'search_term_svd_26', 'search_term_svd_27', 'search_term_svd_28', 'search_term_svd_29', 'search_term_svd_30', 'search_term_svd_31', 'search_term_svd_32', 'search_term_svd_33', 'search_term_svd_34', 'search_term_svd_35', 'search_term_svd_36', 'search_term_svd_37', 'search_term_svd_38', 'search_term_svd_39', 'search_term_svd_40', 'search_term_svd_41', 'search_term_svd_42', 'search_term_svd_43', 'search_term_svd_44', 'search_term_svd_45', 'search_term_svd_46', 'search_term_svd_47', 'search_term_svd_48', 'search_term_svd_49', 'product_title_svd_0', 'product_title_svd_1', 'product_title_svd_2', 'product_title_svd_3', 'product_title_svd_4', 'product_title_svd_5', 'product_title_svd_6', 'product_title_svd_7', 'product_title_svd_8', 'product_title_svd_9', 'product_title_svd_10', 'product_title_svd_11', 'product_title_svd_12', 'product_title_svd_13', 'product_title_svd_14', 'product_title_svd_15', 'product_title_svd_16', 'product_title_svd_17', 'product_title_svd_18', 'product_title_svd_19', 'product_title_svd_20', 'product_title_svd_21', 'product_title_svd_22', 'product_title_svd_23', 'product_title_svd_24', 'product_title_svd_25', 'product_title_svd_26', 'product_title_svd_27', 'product_title_svd_28', 'product_title_svd_29', 'product_title_svd_30', 'product_title_svd_31', 'product_title_svd_32', 'product_title_svd_33', 'product_title_svd_34', 'product_title_svd_35', 'product_title_svd_36', 'product_title_svd_37', 'product_title_svd_38', 'product_title_svd_39', 'product_title_svd_40', 'product_title_svd_41', 'product_title_svd_42', 'product_title_svd_43', 'product_title_svd_44', 'product_title_svd_45', 'product_title_svd_46', 'product_title_svd_47', 'product_title_svd_48', 'product_title_svd_49', 'len_of_query', 'len_of_title', 'len_of_description', 'word_in_title', 'word_in_description']
    best_feats_r = ['averagematch', 'w2v_bestsim', 'cosine', 'product_uid', 'difflibratio', 'w2v_lastsim', 'last_sim', 'S_query', 'checksynonyma', 'S_title', 'query_desc_ratio', 'product_title_svd_0', 'euclidean', 'cityblock', 'w2v_firstsim', 'product_title_svd_46', 'product_title_svd_49', 'product_title_svd_48', 'product_title_svd_45', 'product_title_svd_38', 'product_title_svd_41', 'product_title_svd_31', 'product_title_svd_2', 'product_title_svd_17', 'product_title_svd_30', 'product_title_svd_42', 'product_title_svd_22', 'product_title_svd_34', 'product_title_svd_40', 'product_title_svd_44', 'product_title_svd_39', 'product_title_svd_47', 'product_title_svd_21', 'product_title_svd_43', 'product_title_svd_36', 'search_term', 'product_title_svd_26', 'product_title_svd_28', 'product_title_svd_33', 'w2v_avgsim', 'product_title_svd_8', 'product_title_svd_24', 'product_title_svd_16', 'product_title_svd_32', 'product_title_svd_37', 'product_title_svd_29', 'product_title_svd_13', 'product_title_svd_35', 'len_of_description', 'desc_length', 'product_title_svd_15', 'product_title_svd_20', 'product_title_svd_27', 'product_title_svd_18', 'product_title_svd_23', 'product_title_svd_7', 'product_title_svd_25', 'product_title_svd_19', 'search_term_svd_48', 'search_term_svd_49', 'w2v_totalsim', 'product_title_svd_14', 'product_title_svd_4', 'product_title_svd_11', 'product_title_svd_9', 'product_title_svd_5', 'product_title_svd_12', 'product_title_svd_10', 'product_title_svd_6', 'product_title_svd_1', 'first_sim', 'search_term_svd_44', 'product_title_svd_3', 'search_term_svd_47', 'search_term_svd_0', 'search_term_svd_46', 'search_term_svd_45', 'search_term_svd_42', 'query_title_ratio', 'search_term_svd_40', 'search_term_svd_36', 'search_term_svd_38', 'search_term_svd_31', 'search_term_svd_41', 'search_term_svd_18', 'search_term_svd_23', 'search_term_svd_34', 'search_term_svd_32', 'search_term_svd_37', 'search_term_svd_39', 'search_term_svd_43', 'search_term_svd_2', 'search_term_svd_33', 'search_term_svd_1', 'search_term_svd_30', 'search_term_svd_35', 'search_term_svd_24', 'search_term_svd_4', 'search_term_svd_5', 'search_term_svd_22', 'search_term_svd_15', 'search_term_svd_16', 'search_term_svd_3', 'search_term_svd_19', 'search_term_svd_7', 'search_term_svd_17', 'search_term_svd_29', 'search_term_svd_28', 'search_term_svd_25', 'search_term_svd_8', 'search_term_svd_14', 'search_term_svd_20', 'search_term_svd_12', 'search_term_svd_10', 'search_term_svd_27', 'search_term_svd_26', 'search_term_svd_9', 'search_term_svd_11', 'search_term_svd_21', 'search_term_svd_13', 'query_length', 'search_term_svd_6', 'len_of_title', 'title_length', 'len_of_query', 'word_in_description', 'word_in_title', 'hamming']
    best_feats_r = ['averagematch', 'w2v_bestsim', 'cosine', 'difflibratio', 'n_positions', 'product_uid', 'w2v_lastsim', 'last_sim', 'checksynonyma', 'S_query', 'S_title', 'euclidean', 'search_term', 'product_title_svd_0', 'query_desc_ratio', 'cityblock', 'product_title_svd_45', 'product_title_svd_46', 'product_title_svd_48', 'product_title_svd_31', 'product_title_svd_21', 'product_title_svd_49', 'product_title_svd_39', 'product_title_svd_44', 'product_title_svd_42', 'product_title_svd_35', 'product_title_svd_43', 'product_title_svd_36', 'product_title_svd_22', 'product_title_svd_34', 'product_title_svd_41', 'product_title_svd_47', 'product_title_svd_38', 'product_title_svd_33', 'product_title_svd_16', 'MFG.Brand.Name', 'product_title_svd_17', 'product_title_svd_32', 'product_title_svd_28', 'len_of_description', 'product_title_svd_30', 'product_title_svd_40', 'product_title_svd_29', 'product_title_svd_37', 'w2v_avgsim', 'product_title_svd_15', 'search_term_svd_0', 'product_title_svd_2', 'product_title_svd_24', 'desc_length', 'w2v_totalsim', 'product_title_svd_7', 'product_title_svd_9', 'product_title_svd_27', 'product_title_svd_23', 'product_title_svd_25', 'product_title_svd_11', 'product_title_svd_13', 'product_title_svd_8', 'product_title_svd_26', 'product_title_svd_20', 'product_title_svd_18', 'product_title_svd_12', 'product_title_svd_10', 'product_title_svd_14', 'product_title_svd_19', 'product_title_svd_1', 'product_title_svd_4', 'search_term_svd_1', 'product_title_svd_6', 'product_title_svd_3', 'product_title_svd_5', 'search_term_svd_49', 'search_term_svd_47', 'w2v_firstsim', 'search_term_svd_44', 'search_term_svd_40', 'search_term_svd_45', 'search_term_svd_46', 'query_title_ratio', 'search_term_svd_48', 'search_term_svd_34', 'search_term_svd_37', 'search_term_svd_42', 'search_term_svd_31', 'search_term_svd_43', 'search_term_svd_8', 'search_term_svd_22', 'search_term_svd_41', 'search_term_svd_19', 'search_term_svd_39', 'search_term_svd_36', 'search_term_svd_5', 'search_term_svd_33', 'search_term_svd_18', 'search_term_svd_17', 'search_term_svd_12', 'search_term_svd_38', 'search_term_svd_15', 'search_term_svd_30', 'search_term_svd_7', 'search_term_svd_20', 'search_term_svd_35', 'search_term_svd_14', 'search_term_svd_28', 'search_term_svd_29', 'search_term_svd_2', 'search_term_svd_11', 'search_term_svd_24', 'search_term_svd_25', 'search_term_svd_21', 'search_term_svd_23', 'search_term_svd_32', 'search_term_svd_13', 'search_term_svd_27', 'search_term_svd_16', 'search_term_svd_26', 'search_term_svd_3', 'search_term_svd_10', 'search_term_svd_4', 'word_in_description', 'search_term_svd_9', 'query_length', 'search_term_svd_6', 'first_sim', 'word_in_title', 'len_of_title', 'title_length', 'len_of_query', 'hamming']

    #attribute_list = [u'MFG Brand Name',u'Material',u'Product Width (in.)',u'Color Family',u'Product Height (in.)',u'Product Depth (in.)',u'Product Weight (lb.)',u'Color/Finish',u'Certifications and Listings']
    #attribute_list = [u'MFG Brand Name',u'Material',u'Bullet02',u'Bullet03',u'Bullet04',u'Bullet01',u'Bullet05']
    attribute_list = [u'MFG Brand Name']


    quickload = './data/store_db1b.h5'
    seed = 42
    nsamples =-1
    holdout = 'holdout_chrisslz.csv'#'Xval_id.csv' #"heldout_anttip.csv" #'Xval_id.csv'  #anttip 0.446 versus 0.455
    dropFeatures = ['product_title','product_description','product_info']#+['product_uid'] minus 0.003
    merge_product_infos = True#[u'product_title',u'product_description'] + attribute_list
    keepFeatures = None#best_feats_r[:110]#best30
    dummy_encoding = None
    use_new_spellchecker = False
    labelEncode = ['search_term']+attribute_list
    removeRare_freq = None#38
    oneHotenc = None#['search_term']
    createVerticalFeatures = False
    createVerticalFeatures_new = False
    logtransform = None
    cleanData = True#'load'
    stemmData =  None #'load'
    createCommonWords = True
    useAttributes = attribute_list
    doTFIDF = None#['search_term','product_title']#['search_term','product_info'] # 'load' #['search_term','product_title']# ['search_term','product_title','material','brand']
    n_components = [50,50]
    computeSim = {'columns': ['product_info','search_term'],'reducer':None,'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')}
    #computeSim = {'columns': ['product_info','search_term'],'reducer':MiniBatchKMeans(n_clusters=250),'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')} # RMSE=0.476 vs 0.471 (SVD)

    #{'columns': ['product_info','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')}
    #computeSim = {'columns': ['product_title','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='char',ngram_range=(2, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words)}
    #computeSim = {'columns': ['product_info','search_term'],'doSVD': None, 'vectorizer': TfidfVectorizer( max_features=3500, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')}
    #computeSim = None
    computeAddFeates = False#'load'#True
    computeAddFeates_new = False#'load'#True
    word2vecFeates = False#'load'#True
    word2vecFeates_new = False
    removeCorr = False
    spellchecker = True
    query_correction = False
    """

    quickload = False#'./data/store.h5'
    seed = 42
    nsamples = -1
    holdout = 'Xval_id.csv'# "heldout_anttip.csv" #'Xvald_id.csv'
    dropFeatures = ['product_title','product_description','product_info']#+ attribute_list #['product_uid']
    merge_product_infos = True#[u'product_title',u'product_description'] + attribute_list
    keepFeatures = None#best_feats_r[:110]#best30
    dummy_encoding = None
    labelEncode = ['search_term']#+attribute_list
    removeRare_freq = None#38
    oneHotenc = None#['search_term']
    logtransform = None
    cleanData = None#'load'
    stemmData =  None #'load'
    createCommonWords = True
    useAttributes = None
    doTFIDF = None#['search_term','product_title']#['search_term','product_info'] # 'load' #['search_term','product_title']# ['search_term','product_title','material','brand']
    n_components = [50,50]
    computeSim = None#{'columns': ['product_info','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')}
    #{'columns': ['product_info','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')}
    #computeSim = {'columns': ['product_title','search_term'],'doSVD': 250, 'vectorizer': TfidfVectorizer(min_df=10,  max_features=None, strip_accents='unicode', analyzer='char',ngram_range=(2, 3), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words)}
    #computeSim = {'columns': ['product_info','search_term'],'doSVD': None, 'vectorizer': TfidfVectorizer( max_features=3500, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')}
    #computeSim = None
    computeAddFeates = None#'load'#True
    computeAddFeates_new = None#'load'#True
    word2vecFeates = None#'load'#True
    word2vecFeates_new = False
    removeCorr = False
    spellchecker = False
    query_correction = False
    createVerticalFeatures = False
    """

    Xtest, Xtrain, ytrain, idx, sample_weight, Xval, yval = prepareDataset(quickload=quickload, seed=seed,
                                                                           nsamples=nsamples, holdout=holdout,
                                                                           keepFeatures=keepFeatures,
                                                                           dropFeatures=dropFeatures,
                                                                           use_new_spellchecker=use_new_spellchecker,
                                                                           dummy_encoding=dummy_encoding,
                                                                           labelEncode=labelEncode, oneHotenc=oneHotenc,
                                                                           removeRare_freq=removeRare_freq,
                                                                           logtransform=logtransform,
                                                                           stemmData=stemmData,
                                                                           createCommonWords=createCommonWords,
                                                                           useAttributes=useAttributes, doTFIDF=doTFIDF,
                                                                           computeSim=computeSim,
                                                                           n_components=n_components,
                                                                           cleanData=cleanData,
                                                                           merge_product_infos = merge_product_infos,
                                                                           computeAddFeates=computeAddFeates,
                                                                           computeAddFeates_new=computeAddFeates_new,
                                                                           word2vecFeates=word2vecFeates,
                                                                           word2vecFeates_new = word2vecFeates_new,
                                                                           spellchecker=spellchecker,
                                                                           removeCorr=removeCorr,
                                                                           createVerticalFeatures=createVerticalFeatures,
                                                                           createVerticalFeatures_new=createVerticalFeatures_new,
                                                                           query_correction=query_correction)
    print list(Xtrain.columns)
    #interact_analysis(Xtrain)
    #model = sf.RandomForest(n_estimators=120,mtry=Xtrain.shape[1]/2,node_size=5,max_depth=12,n_jobs=2,verbose_level=0)
    #model = Pipeline([('scaler', StandardScaler()), ('model',ross1)])
    #model = RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=Xtrain.shape[1]/3,oob_score=False)
    #model = XgboostRegressor(n_estimators=400,learning_rate=0.025,max_depth=10, NA=0,subsample=.75,colsample_bytree=0.75,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = XgboostRegressor(n_estimators=200,learning_rate=0.05,max_depth=10, NA=0,subsample=.75,colsample_bytree=0.75,min_child_weight=5,n_jobs=4,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = BaggingRegressor(model,n_estimators=30, max_samples=1.0, max_features=0.9, bootstrap=True) #RMSE0.461
    #model = KernelRidge(alpha=1,kernel='linear',gamma=None)
    #model = RandomForestRegressor(n_estimators = 500, n_jobs = -1, verbose = 1)
    #model = BaggingRegressor(RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0), n_estimators=45, max_samples=0.1, random_state=25)
    #model = ExtraTreesRegressor(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=3*Xtrain.shape[1]/4)
    #model = KNeighbors(n_neighbors=20)
    #model = LinearRegression()
    #model  = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=20,learning_rate=0.0001,validation_split=0.2,batch_size=512,verbose=1,activation='sigmoid', layers=[256,256], dropout=[0.1,0.1],loss='mse') # best

    #model  = KerasNN(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=20,learning_rate=0.0001,validation_split=0.0,batch_size=512,verbose=1,activation='sigmoid', layers=[256,256], dropout=[0.1,0.0],loss='mse')
    #model = Pipeline([('scaler', StandardScaler()), ('nn',model)])
    #model = BaggingRegressor(model,n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=False) #RMSE0.461
    #model = Pipeline([('scaler', RobustScaler()), ('nn',model)])

    #model = KNeighborsRegressor(n_neighbors=10)

    #cv = LabelKFold(Xtrain['product_uid'], n_folds=8)
    cv_labels = pd.Series.from_csv('./data/labels_for_cv.csv')
    cv = LabelKFold(cv_labels, n_folds=8)
    #cv = LabelShuffleSplit(Xtrain['search_term'],n_iter=8, test_size= 0.2 )
    #cv = KFold(Xtrain.shape[0],8,shuffle=True)
    #cv = KLabelFolds(Xtrain['search_term'], n_folds=8, repeats=1)
    scoring_func = make_scorer(root_mean_squared_error, greater_is_better=False)
    # cv = KFold(X.shape[0], n_folds=folds,shuffle=True)
    #cv = StratifiedShuffleSplit(ytrain,2)
    #scoring_func = roc_auc_score
    #print df.printSummary()
    #parameters = {'n_estimators':[2000,4000],'max_depth':[20],'learning_rate':[0.01,0.001],'subsample':[0.5],'colsample_bytree':[0.9],'min_child_weight':[5]}
    #parameters = {'n_estimators':[500],'min_samples_leaf':[5,8,10],'max_features':[100]}
    #parameters = {'nn__nb_epoch':[20,40],'nn__learning_rate':[0.01], 'nn__dropout':[[0.1]*2,[0.05]*2],'nn__layers':[[256]*2,[512]*2],'nn__batch_size':[64]}
    #parameters={'n_neighbors':[20,25,30]}
    #model = makeGridSearch(model, Xtrain, ytrain, n_jobs=2, refit=True, cv=cv, scoring=scoring_func,parameters=parameters, random_iter=-1)

    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    print Xtrain.shape
    print model
    buildModel(model,Xtrain,ytrain,cv=cv, scoring=scoring_func, n_jobs=2,trainFull=False,verbose=True)
    #analyzeLearningCurve(model, Xtrain, ytrain, cv=cv, score_func='roc_auc')
    #buildXvalModel(model,Xtrain,ytrain,sample_weight=None,class_names=None,refit=False,cv=cv)

    print "Evaluation data set..."
    model.fit(Xtrain,ytrain)
    yval_pred = model.predict(Xval)
    #yval_pred = post_process(yval_pred,yval,sigma=0.1,k=0.5)

    print " Eval-score: %5.4f"%(root_mean_squared_error(yval,np.clip(yval_pred,1.0,3.0)))

    print "Training the final model (incl. Xval.)"
    Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    model.fit(Xtrain,ytrain)

    makePredictions(model,Xtest,idx=idx, filename='./submissions/sub04032016c.csv')

    plt.show()
    print("Model building done in %fs" % (time() - t0))
