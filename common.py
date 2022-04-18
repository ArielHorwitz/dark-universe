# industry module - chemistry, ship modules and components

from nutil import *
from nutil import lists as nlists
import collections
import math
import numpy as np

PLANK = 10**-5
PROXIMITY_DISTANCE = 10**-1


class Genesis(dict):
    SCALES = {
        '0': {'mass': 10**30, 'orbit': 0},
        '1': {'mass': 10**18, 'orbit': 10**6},
        '2': {'mass': 10**12, 'orbit': 10**5},
        '3': {'mass': 10**9, 'orbit': 10**4},
        '4': {'mass': 10**7, 'orbit': 10**3},
        '5': {'mass': 10**4, 'orbit': 10**2},
        # '6': {'mass': 10**3, 'orbit': 10**1},
        }
    def __init__(self, seed, fractal_children, scale=10**8, fractal_scale=15):
        self.total_nodes = 0
        self.seed = Seed(seed)
        self.fractal_children = fractal_children
        self.scale = scale
        self.fractal_scale = fractal_scale
        # self._names = copy.deepcopy(lists.CELESTIAL_NAMES)
        self.tree = {
            'index': [],
            'location': np.zeros(2),
            'findex': 0,
            'children': [],
            }
        self.nodes = []
        ccount = max(0, self.seed.randint(self.fractal_children[0][0], self.fractal_children[0][1]+1))
        for cindex in range(ccount):
            self.tree['children'].append(self.make_node(self.tree, cindex))
        self.nodes = [*reversed(self.nodes)]

    @classmethod
    def min_loc_sort(cls, location, siblings):
        if len(siblings) == 0:
            return 0
        min_dist = min([vmag(location - sib['location']) for sib in siblings])
        return min_dist

    def make_node(self, parent, cindex):
        # name = self.seed.pop(self._names)
        index = parent['index'] + [cindex]
        findex = len(index)
        max_radius = Genesis.SCALES[str(findex)]['orbit']
        locations = []
        location_attempts = len(parent['children'])+1
        for attempt in range(location_attempts):
            new_loc = parent['location'] + angleradius_coords(
                self.seed.randfloat(max_radius*0.2, max_radius),
                self.seed.randfloat(360))
            locations.append(new_loc)
        location = sorted(locations, key=lambda x, siblings=parent['children']: -Genesis.min_loc_sort(x, siblings))[0]
        node = {
            'index': index,
            'location': location,
            'children': [],
            }
        if findex < len(self.fractal_children):
            ccount = self.seed.randint(self.fractal_children[findex][0], self.fractal_children[findex][1]+1)
            for cindex_ in range(ccount):
                node['children'].append(self.make_node(node, cindex_))
        self.total_nodes += 1
        self.nodes.append((node['index'], node['location']))
        return node

    def random_coord(self, max_range):
        return np.array([self.seed.randfloat(-max_range, max_range) for _ in range(2)])

    def get_fractal_scale(self, findex):
        # return self.scale * (self.fractal_scale**-(findex-1))
        return


class Names:
    celestial = nlists.CELESTIAL_NAMES
    ship_prefixes = [
        'XSS',  # 'ISS', 'IFS', 'XSS', 'XSP',
        ]
    ship_class_names = [
        'Segwit', 'ProofOfWork', 'BIP', 'Whitepaper', 'ECDSA', 'Halvening',
        'Multisig', 'ZeroConf', 'Node',
        'Degree', 'Shortside', 'Longside', 'Aperture', 'Iris', 'Resolve', 'Control', 'Film',
        'Prism', 'Ibis', 'CMOS', 'Register',
        'Rava', 'Vaatu', 'Fire', 'Earth', 'Water', 'Air',

        ]
    ship_names = [
        'Nakamoto', 'Satoshi', 'Van-der-Laan', 'MarcoFalke', 'FanQuake', 'Andresen',
        'Maxwell', 'Sipa', 'Rosenfeld', 'Andreas', 'Ivgi', 'Saylor', 'Meister', 'Weatherman', 'Chewey',
        'Szabo', 'Finney',
        'Toph', 'Sokka', 'Katara', 'Aang', 'Iroh', 'Zuko', 'Appa', 'Momo',
        'Korra', 'Bolin', 'Mako', 'Asami', 'Zhu Lee', 'Varic', 'Naga', 'Pabu',
        'Wan', 'Roku', 'Kyoshi', 'Kuruk', 'Yangchen', 'Tenzin', 'Zahir', 'Boomie', 'Ozai', 'Lin',

        ]
    ship_adjectives = [
        'vengeful', 'reliable', 'resilient', 'grand', 'brilliant', 'colorful', 'colossal', 'proud',
        'brave', 'drab', 'faithful', 'elegant', 'thoughtful', 'polite', 'lively', 'victorious',
        'witty', 'fierce', 'mysterious',
        *nlists.COLOR_NAMES[2:],
        ]
    rock_prefixes = '¿╬‡§ø¤'

    @classmethod
    def get_class_name(cls, seed=None):
        seed = Seed() if seed is None else seed
        return f'{seed.choice(cls.ship_class_names)}-class'

    @classmethod
    def get_ship_name(cls, seed=None):
        seed = Seed() if seed is None else seed
        pref = seed.choice(cls.ship_prefixes)
        name_adj = seed.choice(cls.ship_adjectives)
        name_base = seed.choice(cls.ship_names)
        return f'{pref}. {name_adj.capitalize()} {name_base}'


class Element:
    S = nlists.LETTERS
    SR = ''.join(reversed(nlists.LETTERS))
    NAMES = [
        'Aradium',
        'Bohirium',
        'Corbulium',
        'Deufarium',
        'Eccenium',
        'Ferrium',
        'Geovium',
        'Hardium',
        'Iodium',
        'Jinium',
        'Kovelium',
        'Laminium',
        'Marinium',
        'Nordinium',
        'Oltarium',
        'Postivium',
        'Quarzium',
        'Rafolium',
        'Stolium',
        'Tilium',
        'Uvalium',
        'Varinium',
        'Weavium',
        'Xinium',
        'Yedium',
        'Zorium',
        ]
    PROPERTIES = [
        'Energy',
        'Hardness',
        'Flexibility',
        'Viscosity',
        'Reactivity',
        'Conductivity',
        'Magnetism',
        'Ionization',
        'Phase',
        # 'Vibration',
        # 'Stiffness',
        # 'Density',
        ]
    NUMBERED_PROPS = [f'{pi}. {pn}' for pi, pn in [*enumerate(PROPERTIES)][2:]]
    ECOUNT = len(NAMES)
    PCOUNT = len(PROPERTIES)
    PCAP = 0.8
    PMAG = 5
    RARITY_CURVE = 0.7  # Rarest element has roughly 2.8 percent chance of spawning compared to most common element
    RARITY_FACTOR = 0.8
    BASE_RARITY_CURVE = 0.2
    MAX_RARITY_CURVE = 3

    @classmethod
    def find_ename(cls, x):
        """Looks at the first letter of the query (x) and finds the respective element name. Defaults to the first element."""
        x = x[0].lower()
        for ename in cls.NAMES:
            if x == ename[0].lower():
                return ename
        return cls.NAMES[0]

    @classmethod
    def e2i(cls, x):
        if isinstance(x, dict):
            return {cls.NAMES.index(k): v for k, v in x.items()}
        return cls.NAMES.index(x)

    @classmethod
    def i2e(cls, x):
        if isinstance(x, dict):
            return {cls.NAMES[k]: v for k, v in x.items()}
        return cls.NAMES[x]

    @staticmethod
    def spawn_chance(x, variance, mean=0):
        """
        Sample is a value between 0 and 1.

        Relative spawn chance in percent per element (y) per variance (x):
        [  Element ,    0.200,    0.250,    0.300,    0.330,    0.350,    0.400,    0.450,    0.500,    0.750,    1.000,    3.000,    5.000]
        ====================================================================================================================================
        [     0.0  ,  100.000,  100.000,  100.000,  100.000,  100.000,  100.000,  100.000,  100.000,  100.000,  100.000,  100.000,  100.000]
        [     1.000,   98.019,   98.728,   99.115,   99.268,   99.349,   99.501,   99.605,   99.680,   99.857,   99.920,   99.991,   99.996]
        [     2.000,   92.311,   95.008,   96.506,   97.104,   97.421,   98.019,   98.432,   98.728,   99.432,   99.680,   99.964,   99.987]
        [     3.000,   83.527,   89.118,   92.311,   93.602,   94.291,   95.599,   96.506,   97.161,   98.728,   99.282,   99.920,   99.971]
        [     4.000,   72.614,   81.481,   86.742,   88.910,   90.078,   92.311,   93.874,   95.008,   97.750,   98.728,   99.857,   99.948]
        [     5.000,   60.653,   72.614,   80.073,   83.222,   84.936,   88.249,   90.595,   92.311,   96.506,   98.019,   99.778,   99.920]
        [     6.000,   48.675,   63.077,   72.614,   76.761,   79.049,   83.527,   86.742,   89.118,   95.008,   97.161,   99.680,   99.884]
        [     7.000,   37.531,   53.408,   64.690,   69.770,   72.614,   78.270,   82.400,   85.487,   93.268,   96.155,   99.565,   99.843]
        [     8.000,   27.803,   44.078,   56.615,   62.490,   65.838,   72.614,   77.659,   81.481,   91.299,   95.008,   99.432,   99.795]
        [     9.000,   19.789,   35.458,   48.675,   55.153,   58.920,   66.697,   72.614,   77.166,   89.118,   93.725,   99.282,   99.741]
        [    10.000,   13.533,   27.803,   41.111,   47.968,   52.045,   60.653,   67.363,   72.614,   86.742,   92.311,   99.115,   99.680]
        [    11.000,    8.892,   21.250,   34.110,   41.111,   45.375,   54.607,   62.000,   67.895,   84.190,   90.773,   98.930,   99.613]
        [    12.000,    5.613,   15.831,   27.803,   34.720,   39.046,   48.675,   56.615,   63.077,   81.481,   89.118,   98.728,   99.540]
        [    13.000,    3.404,   11.495,   22.263,   28.894,   33.165,   42.955,   51.291,   58.228,   78.634,   87.354,   98.509,   99.460]
        [    14.000,    1.984,    8.136,   17.513,   23.696,   27.803,   37.531,   46.101,   53.408,   75.672,   85.487,   98.272,   99.374]
        [    15.000,    1.110,    5.613,   13.533,   19.149,   23.006,   32.465,   41.111,   48.675,   72.614,   83.527,   98.019,   99.282]
        [    16.000,    0.597,    3.774,   10.273,   15.249,   18.790,   27.803,   36.372,   44.078,   69.482,   81.481,   97.750,   99.184]
        [    17.000,    0.308,    2.474,    7.662,   11.966,   15.147,   23.574,   31.926,   39.661,   66.297,   79.358,   97.463,   99.079]
        [    18.000,    0.153,    1.580,    5.613,    9.253,   12.052,   19.789,   27.803,   35.458,   63.077,   77.166,   97.161,   98.968]
        [    19.000,  7.3e-02,    0.984,    4.040,    7.051,    9.465,   16.447,   24.022,   31.499,   59.844,   74.916,   96.842,   98.851]
        [    20.000,  3.4e-02,    0.597,    2.856,    5.294,    7.336,   13.533,   20.592,   27.803,   56.615,   72.614,   96.506,   98.728]
        [    21.000,  1.5e-02,    0.353,    1.984,    3.917,    5.613,   11.025,   17.513,   24.385,   53.408,   70.271,   96.155,   98.598]
        [    22.000,  6.3e-03,    0.203,    1.353,    2.856,    4.239,    8.892,   14.776,   21.250,   50.240,   67.895,   95.789,   98.463]
        [    23.000,  2.5e-03,    0.114,    0.907,    2.052,    3.159,    7.100,   12.370,   18.400,   47.125,   65.494,   95.406,   98.321]
        [    24.000,  9.9e-04,  6.3e-02,    0.597,    1.453,    2.324,    5.613,   10.273,   15.831,   44.078,   63.077,   95.008,   98.173]
        [    25.000,  3.7e-04,  3.4e-02,    0.386,    1.013,    1.687,    4.393,    8.465,   13.533,   41.111,   60.653,   94.595,   98.019]
        """
        # return 1/(variance*math.sqrt(2*math.pi)) * math.e**(-(x-mean)**2 / (2*variance**2))
        zero = 1/(variance*math.sqrt(2*math.pi)) * math.e**(-(0-mean)**2 / (2*variance**2))
        return 1/(variance*math.sqrt(2*math.pi)) * math.e**(-(x-mean)**2 / (2*variance**2)) / zero

    @staticmethod
    def gen_elements(seed, bias=0):
        variance = Element.BASE_RARITY_CURVE+(bias*(Element.MAX_RARITY_CURVE-Element.BASE_RARITY_CURVE))
        s = Seed(seed)
        elements = {}
        for eindex in range(Element.ECOUNT):
            present_prob = Element.spawn_chance(eindex/Element.ECOUNT, variance)

        for eindex in range(Element.ECOUNT):
            present_prob = Element.spawn_chance(eindex/Element.ECOUNT, variance)
            if s.r < present_prob:
                cap = s.r
                access = minmax(0.05, 1, (1/(1-s.r)**0.25)-1)
            else:
                cap = 0
                access = 0
            elements[eindex] = {
                'capacity': cap,
                'accessibility': access,
                'present': present_prob,
                }
        return elements

    @staticmethod
    def PeriodicTable():
        ptable = {eindex: {pindex+1: None for pindex in range(Element.PCOUNT)} for eindex in range(Element.ECOUNT)}
        for eindex in ptable:
            for pindex in ptable[eindex]:
                max_columns = pindex
                max_rows = Element.ECOUNT // max_columns + 1
                r, c = Element.row_col(eindex, max_columns)
                # Set values between 0 and PCAP
                # rv = (r+1) / (max_rows) * Element.PCAP
                # cv = (c+1) / (max_columns) * Element.PCAP
                # Considering to set values as 2**n where n is between 0 and PMAG for balance purposes
                rv = 2**((r+1)/max_rows*Element.PMAG) / 2**Element.PMAG
                cv = 2**((c+1)/max_columns*Element.PMAG) / 2**Element.PMAG
                ptable[eindex][pindex] = (rv, cv)
        return ptable

    @staticmethod
    def row_col(eindex, prop):
        return eindex // prop, eindex % prop

    @staticmethod
    def spawn_elements(seed, bias, sample_size=10):
        # variance = Element.BASE_RARITY_CURVE+(bias*(Element.MAX_RARITY_CURVE-Element.BASE_RARITY_CURVE))
        variance = bias
        prob = []
        for ei in range(Element.ECOUNT):
            prob.append(Element.spawn_chance(ei/Element.ECOUNT, variance))
        prob_sum = sum(prob)
        wprob = []
        for rawp in prob:
            wprob.append(rawp/prob_sum)

        # Start sampling
        s = Seed(seed)
        spawns = []
        for si in range(sample_size):
            spawns.append(find_by_weight(s.r, weights=wprob))
        return spawns


class Sensing:
    RADIUS_FACTOR = 1.5

    @classmethod
    def volume_per_radius(cls, radius):
        return (4*math.pi/3) * (radius**3)

    @classmethod
    def surface_area_per_radius(cls, radius):
        return (4*math.pi) * (radius**2)

    @classmethod
    def radius_per_volume(cls, volume):
        """Volume = (4*pi/3) * radius**3"""
        return (volume / (4/3*math.pi))**(1/3)

    @classmethod
    def radius_per_surface_area(cls, sa):
        """Surface Area = (4*pi) * radius**2"""
        return (sa / (4*math.pi))**(1/2)

    @classmethod
    def sensor_range(cls, mass, sens):
        """
        Assume sensitivity corresponds to the surface area of the sphere we can sense (the sensor "envelope"). Assume mass with a constant density (equal to the volume of the body being detected). Assume sensor detects the cross section of the body being detected, equal to the area of the circle given by the radius of the sphere of said body (equal to the projection of said sphere).
        The sensor range corresponds to the radius of the sphere of sensing per cross section of the mass.
        """
        return 10**Sensing.RADIUS_FACTOR * Sensing.radius_per_surface_area(sens) * Sensing.radius_per_volume(mass)

    @classmethod
    def test_sensor_range(cls):
        masses = np.linspace(1, 8, 19)
        senses = np.linspace(0.1, 4, 10)
        blank = '-'*10
        blank_line = [blank]+[blank for _ in senses]
        sat = {
            0: [0]+[10**(s*10) for s in senses],
            blank: blank_line,
            1: [0]+[Sensing.radius_per_surface_area(10**(s*10)) for s in senses],
            }
        vt = {
            0: [0]+[10**(s*10) for s in senses],
            blank: blank_line,
            1: [0]+[Sensing.radius_per_volume(10**(s*10)) for s in senses],
            }
        st = {
            0: [0]+[10**j for j in senses],
            blank: blank_line,
            }
        for i in masses:
            st[i] = [10**i]
            for j in senses:
                st[i].append(Sensing.sensor_range(10**i, 10**j))

        #
        print(make_title('Surface Area per Volume'))
        for i, r in sat.items():
            print(sjoin((adis(_, precision=1, force_scientific=False) for _ in r), split=', '))
        print(make_title('Radius per Volume'))
        for i, r in vt.items():
            print(sjoin((adis(_, precision=1, force_scientific=False) for _ in r), split=', '))

        print(make_title('Sensor range per Mass * Sens'))
        for i, r in st.items():
            print(sjoin((adis(_, precision=1, force_scientific=False) for _ in r), split=', '))

    @staticmethod
    def old_sensor_range(sensitivity, mass):
        mass_factor = 5
        sens_factor = 1
        total_factor = -1
        total_scale = 3.5
        r = 10**total_factor * (math.log(sensitivity**sens_factor) * math.log(mass**mass_factor))**total_scale
        return r


class Tech(list):
    def __init__(self, seed, max_val=0.5, averaging_factor=3):
        """A class representing values from a seed"""
        self.s = Seed(seed)
        vals = [sum([self.s.r*max_val for rf_ in range(averaging_factor)])/averaging_factor for val_ in range(5)]
        super().__init__(vals)

    @property
    def normal_value(self):
        return self.normal(1) + self.normal(2)

    def normal(self, i):
        return self[i]

    def log(self, i):
        return 1/(1-self[i]) - 1

    def inverted(self, i):
        return 1-self[i]

    def ilog(self, i):
        return 1 - self.log(i)

    def prop(self, i):
        return Seed(self.normal(i)).randint(1, E.PCOUNT)


def angleradius_coords(radius, angle=None):
    angle = angle * math.pi / 180
    return np.array([radius * math.cos(angle), radius * math.sin(angle)])

def find_by_weight(value, weights):
    for i, w in enumerate(weights):
        if value < w:
            return i
        value -= w
    return 0

def cargo_mass(cargo):
    total = 0
    for item, count in cargo.items():
        # For now we only store elements which all weigh exactly 1 each
        assert count >= 0
        total += count
    return total

def vmag(a):
    return np.linalg.norm(a)

E = Element
PT = Element.PeriodicTable()
def ptable(e, p):
    if isinstance(e, str):
        e = Element.NAMES.index(e)
    if isinstance(p, str):
        p = Element.PROPERTIES.index(p)
    return PT[e][p]
