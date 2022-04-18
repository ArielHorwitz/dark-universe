# Components module - component and ship design

from nutil import *
import common


class Technology:
    def __init__(self, element, prop):
        assert isinstance(element, int)
        assert isinstance(prop, int)
        assert 3 <= prop <= 9
        self.element = element
        self.prop = prop
        self.quark, self.boson = common.PT[element][prop]


class BaseComponent:
    STATS = ['mass', 'emissions', 'maint_cost', 'material_cost', 'build_cost']
    MAINTENANCE_FACTOR = 10**-1

    def __init__(self,
                 ctype, mass, material,
                 manufacturing, technology,
                 name=None):
        if isinstance(material, int):
            material = common.E.i2e(material)
        material_index = common.E.e2i(material)
        if name is None:
            name = f'{round(mass)} {material} {ctype} [{manufacturing}, {technology}]'
        self.name = str(name)
        assert mass > 0
        self.mass = mass
        self.material = material
        self.manufacturing = Technology(material_index, manufacturing)
        self.technology = Technology(material_index, technology)
        # All base stats between 0 and 1
        # Emissions
        self.emissions = 1 * self.mass
        # Maintenance is determined by the material such that A costs twice as much as Z
        mcost = ((1 - material_index / (common.E.ECOUNT-1)) * self.mass + self.mass) * BaseComponent.MAINTENANCE_FACTOR
        self.maint_cost = {self.material: mcost}

        # Manufacturing
        matcost = self.mass / (0.25 + 0.75 * self.manufacturing.quark)
        self.material_cost = {self.material: matcost}
        self.build_cost = self.mass / (0.2 + 0.8 * self.manufacturing.boson)

        self.stats = {
            'metadata': {
                'name': self.name,
                'ctype': ctype,
                'material': self.material,
                'manufacturing': manufacturing,
                'technology': technology,
                },
            'mass': self.mass,
            'emissions': self.emissions,
            'maint_cost': self.maint_cost,
            'material_cost': self.material_cost,
            'build_cost': self.build_cost,
            }

    @classmethod
    def combine_base_statblock(cls, components):
        stat_block = {
            'metadata': {
                'spec': [],
                },
            'mass': 0,
            'emissions': 0,
            'maint_cost': {},
            'material_cost': {},
            'build_cost': 0,
            }
        for c_stats in components:
            if 'name' in c_stats['metadata']:
                stat_block['metadata']['spec'].append(c_stats['metadata']['name'])
            elif 'spec' in c_stats['metadata']:
                stat_block['metadata']['spec'].extend(c_stats['metadata']['spec'])
            stat_block['mass'] += c_stats['mass']
            stat_block['emissions'] += c_stats['emissions']
            DictOp.add(stat_block['maint_cost'], c_stats['maint_cost'])
            DictOp.add(stat_block['material_cost'], c_stats['material_cost'])
            stat_block['build_cost'] += c_stats['build_cost']
        return stat_block


class Engineer(BaseComponent):
    ENGINEERING_FACTOR = 10
    EMISSIONS_FACTOR = 1.5
    CTYPE = 'Engineer'
    STATS = ['engineering']

    def __init__(self, *args, **kwargs):
        super().__init__(Engineer.CTYPE, *args, **kwargs)
        engineer = self.mass + self.mass * Engineer.ENGINEERING_FACTOR * self.technology.quark
        emissions = self.mass * Engineer.EMISSIONS_FACTOR / self.technology.boson
        self.stats['engineering'] = engineer
        self.stats['emissions'] += emissions

    @classmethod
    def combine(cls, components):
        statblock = super().combine_base_statblock(components)
        total = 0
        for c_stats in components:
            total += c_stats['engineering']
        statblock = {
            **statblock,
            'engineering': total,
            }
        return statblock


class Cargo(BaseComponent):
    VOLUME_FACTOR = 2
    EMISSIONS_FACTOR = 0
    CTYPE = 'Cargo'
    STATS = ['hold']

    def __init__(self, *args, **kwargs):
        super().__init__(Cargo.CTYPE, *args, **kwargs)
        hold = (self.mass ** Cargo.VOLUME_FACTOR) * self.technology.quark
        emissions = self.mass * Cargo.EMISSIONS_FACTOR / self.technology.boson
        self.stats['hold'] = hold
        self.stats['emissions'] += emissions

    @classmethod
    def combine(cls, components):
        statblock = super().combine_base_statblock(components)
        hold = 0
        for c_stats in components:
            hold += c_stats['hold']
        statblock = {
            **statblock,
            'hold': hold,
            }
        return statblock


class FuelTank(BaseComponent):
    VOLUME_FACTOR = 2
    EMISSIONS_FACTOR = 0
    CTYPE = 'Fuel tank'
    STATS = ['fuel_tank']

    def __init__(self, *args, **kwargs):
        super().__init__(FuelTank.CTYPE, *args, **kwargs)
        tank = (self.mass ** FuelTank.VOLUME_FACTOR) * self.technology.quark
        emissions = self.mass * FuelTank.EMISSIONS_FACTOR / self.technology.boson
        self.stats['fuel_tank'] = tank

    @classmethod
    def combine(cls, components):
        statblock = super().combine_base_statblock(components)
        tank = 0
        for c_stats in components:
            tank += c_stats['fuel_tank']
        statblock = {
            **statblock,
            'fuel_tank': tank,
            }
        return statblock


class Drive(BaseComponent):
    ISP_FACTOR = 10
    THRUST_FACTOR = 10
    EMISSIONS_FACTOR = 5
    CTYPE = 'Drive'
    STATS = ['thrust', 'isp']

    def __init__(self, *args, **kwargs):
        super().__init__(Drive.CTYPE, *args, **kwargs)
        isp = Drive.ISP_FACTOR * self.technology.quark
        thrust = self.mass * Drive.THRUST_FACTOR * self.technology.boson
        emissions = self.mass * Drive.EMISSIONS_FACTOR / self.technology.boson
        self.stats['isp'] = isp
        self.stats['thrust'] = thrust
        self.stats['emissions'] += emissions

    @classmethod
    def combine(cls, components):
        statblock = super().combine_base_statblock(components)
        isps = []
        thrusts = []
        for c_stats in components:
            isps.append(c_stats['isp'])
            thrusts.append(c_stats['thrust'])
        total_isp = 0
        if sum(thrusts) > 0:
            total_isp = weighted_average(isps, thrusts)
        statblock = {
            **statblock,
            'thrust': sum(thrusts),
            'isp': total_isp,
            }
        return statblock


class Miner(BaseComponent):
    MINING_FACTOR = 10**0
    EMISSIONS_FACTOR = 3
    CTYPE = 'Miner'
    STATS = ['mining_capacity']

    def __init__(self, *args, **kwargs):
        super().__init__(Miner.CTYPE, *args, **kwargs)
        mining = self.mass * Miner.MINING_FACTOR * self.technology.quark
        emissions = self.mass * Miner.EMISSIONS_FACTOR / self.technology.boson
        self.stats['mining_capacity'] = mining
        self.stats['emissions'] += emissions

    @classmethod
    def combine(cls, components):
        statblock = super().combine_base_statblock(components)
        total = 0
        for c_stats in components:
            total += c_stats['mining_capacity']
        statblock = {
            **statblock,
            'mining_capacity': total,
            }
        return statblock


class Builder(BaseComponent):
    BUILD_FACTOR = 10**0
    EMISSIONS_FACTOR = 2
    CTYPE = 'Builder'
    STATS = ['build_capacity']

    def __init__(self, *args, **kwargs):
        super().__init__(Builder.CTYPE, *args, **kwargs)
        build = self.mass * Builder.BUILD_FACTOR * self.technology.quark
        emissions = self.mass * Builder.EMISSIONS_FACTOR / self.technology.boson
        self.stats['build_capacity'] = build
        self.stats['emissions'] += emissions

    @classmethod
    def combine(cls, components):
        statblock = super().combine_base_statblock(components)
        total = 0
        for c_stats in components:
            total += c_stats['build_capacity']
        statblock = {
            **statblock,
            'build_capacity': total,
            }
        return statblock


class Sensor(BaseComponent):
    SENSITIVITY_FACTOR = 1
    EMISSIONS_FACTOR = 5
    CTYPE = 'Sensor'
    STATS = ['sensitivity']

    def __init__(self, *args, **kwargs):
        super().__init__(Sensor.CTYPE, *args, **kwargs)
        sens = self.mass * Sensor.SENSITIVITY_FACTOR * self.technology.quark
        emissions = self.mass * Sensor.EMISSIONS_FACTOR / self.technology.boson
        self.stats['sensitivity'] = sens
        self.stats['emissions'] += emissions

    @classmethod
    def combine(cls, components):
        statblock = super().combine_base_statblock(components)
        total_sens = 0
        for c_stats in components:
            total_sens += c_stats['sensitivity']
        statblock = {
            **statblock,
            'sensitivity': total_sens,
            }
        return statblock


class Survey(BaseComponent):
    SURVEY_FACTOR = 1
    EMISSIONS_FACTOR = 2
    CTYPE = 'Survey'
    STATS = ['survey']

    def __init__(self, *args, **kwargs):
        super().__init__(Survey.CTYPE, *args, **kwargs)
        survey = self.mass * Survey.SURVEY_FACTOR * self.technology.quark
        emissions = self.mass * Survey.EMISSIONS_FACTOR / self.technology.boson
        self.stats['survey'] = survey
        self.stats['emissions'] += emissions

    @classmethod
    def combine(cls, components):
        statblock = super().combine_base_statblock(components)
        total_survey = 0
        for c_stats in components:
            total_survey += c_stats['survey']
        statblock = {
            **statblock,
            'survey': total_survey,
            }
        return statblock


class ShipDesign:
    def __init__(self, comps):
        categorized = {ctype: [] for ctype in COMP_TYPES}
        cat_stats = {}
        for comp in comps:
            categorized[comp['metadata']['ctype']].append(comp)
        for ctype, ccls in COMP_TYPES.items():
            cat_stats[ctype] = ccls.combine(categorized[ctype])
        self.stats = self.integrate(cat_stats)

    @classmethod
    def custom(cls, sane_defaults=False, name='Custom ship design', description='Custom design', **kwargs):
        if 'mass' not in kwargs:
            kwargs['mass'] = 1
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {
                'name': name,
                'description': description,
                }
        if sane_defaults:
            if 'engineering' not in kwargs:
                kwargs['engineering'] = kwargs['mass']
            if 'hold' not in kwargs:
                kwargs['hold'] = 100
            if 'fuel_tank' not in kwargs:
                kwargs['fuel_tank'] = 20
        custom_ship = cls([])
        for k, v in kwargs.items():
            if k in custom_ship.stats:
                custom_ship.stats[k] = v
            else:
                print(f'Failed to find stat {k} for custom ship design.')
        return custom_ship

    @classmethod
    def integrate(cls, statblocks):
        assert all([ctype in statblocks for ctype in COMP_TYPES])
        base_block = BaseComponent.combine_base_statblock(statblocks.values())
        for ctype, ccls in COMP_TYPES.items():
            for stat in ccls.STATS:
                base_block[stat] = statblocks[ctype][stat]
        return base_block


COMP_TYPES = {
    'Engineer': Engineer,
    'Cargo': Cargo,
    'Fuel tank': FuelTank,
    'Drive': Drive,
    'Miner': Miner,
    'Builder': Builder,
    'Sensor': Sensor,
    'Survey': Survey,
    }
BASE_STATS = BaseComponent.STATS
SPECIAL_STATS = sum((_.STATS for _ in COMP_TYPES.values()), [])
ALL_STATS = [*BASE_STATS, *SPECIAL_STATS]
STAT_NAMES_PRETTY = {
    'engineering': 'Engineering Cap',
    'hold': 'Cargo hold',
    'fuel_tank': 'Fuel Tank',
    'thrust': 'Thrust',
    'isp': 'Specific Impulse',
    'mining_capacity': 'Mining Cap',
    'build_capacity': 'Build Cap',
    'sensitivity': 'Sensitivity',
    'survey': 'Survey',
    }

CUSTOM_SHIPS = {
    'station': {
        'name': 'Prehistoric Station',
        'description': 'Prehistoric station',
        'stats': ShipDesign.custom(
            metadata={
                'name': 'Prehistoric Station',
                'description': 'Prehistoric station',
                },
            # Base stats
            mass=4000,
            emissions=10000,
            maint_cost={'Aradium': 1},
            material_cost={'Aradium': 30000},
            build_cost=10000,
            # Special stats
            engineering=50000,
            sensitivity=1,
            hold=20000,
            build_capacity=5,
            mining_capacity=5,
            ).stats,
        },
    'miner': {
        'name': 'Prehistoric Miner',
        'description': 'Prehistoric miner',
        'stats': ShipDesign.custom(
            metadata={
                'name': 'Prehistoric Miner',
                'description': 'Prehistoric miner',
                },
            # Base stats
            mass=200,
            emissions=1100,
            maint_cost={'Aradium': 1},
            material_cost={'Aradium': 1000},
            build_cost=200,
            # Special stats
            engineering=200,
            hold=200,
            mining_capacity=5,
            ).stats,
        },
    'explorer': {
        'name': 'Prehistoric Explorer',
        'description': 'Prehistoric explorer',
        'stats': ShipDesign.custom(
            metadata={
                'name': 'Prehistoric Explorer',
                'description': 'Prehistoric explorer',
                },
            # Base stats
            mass=50,
            emissions=400,
            maint_cost={'Aradium': 1},
            material_cost={'Aradium': 1000},
            build_cost=100,
            # Special stats
            engineering=500,
            fuel_tank=50,
            thrust=10,
            isp=0.2,
            sensitivity=0.2,
            survey=1,
            ).stats,
        },
    'hauler': {
        'name': 'Prehistoric Hauler',
        'description': 'Prehistoric hauler',
        'stats': ShipDesign.custom(
            metadata={
                'name': 'Prehistoric Hauler',
                'description': 'Prehistoric hauler',
                },
            # Base stats
            mass=200,
            emissions=500,
            maint_cost={'Aradium': 1},
            material_cost={'Aradium': 5000},
            build_cost=500,
            # Special stats
            engineering=500,
            hold=800,
            fuel_tank=50,
            thrust=50,
            isp=0.2,
            ).stats,
        },
    'dev': {
        'name': 'Dev Ship',
        'description': 'Omnipotenet ship used for development',
        'stats': ShipDesign.custom(
            metadata={
                'name': 'Dev Ship',
                'description': 'Omnipotenet ship used for development',
                },
            # Base stats
            mass=10,
            emissions=1,
            maint_cost={'Aradium': 1},
            material_cost={'Aradium': 1},
            build_cost=1,
            # Special stats
            engineering=10**5,
            hold=10**8,
            fuel_tank=20,
            thrust=10**5,
            isp=10**5,
            build_capacity=10**4,
            mining_capacity=10**4,
            sensitivity=1,
            survey=10**5,
            ).stats,
        },
}

STARTING_SHIPS_STATS = [CUSTOM_SHIPS[_] for _ in (
    # 'station',
    # 'miner',
    # 'miner',
    # 'explorer',
    # 'hauler',
    'dev',
    'dev',
    'dev',
    )]
