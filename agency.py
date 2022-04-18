# Agency module - ship bridge and orders related logic

import math
import numpy as np

from nutil import *
from nutil import lists
import common
import components



STARTING_ARADIUM = 1000

# PLAYER AGENCY

class Player:
    def __init__(self, uni, pid, seed, name, spawn_buid, ships=None,
                 technologies=None, component_designs=None, ship_designs=None,
                 visible_bodies=None, visible_bodies_updated=None,
                 starting_station=None,
                 generate_starting_designs=True):
        self.uni = uni
        self.seed = seed
        self.name = name
        self.pid = pid
        self.spawn_buid = spawn_buid
        self.rock_database = {}
        self.notifications = []
        self.ships = set() if ships is None else ships
        self.__default_technology_names = ['Prehistoric', 'Conventional', 'Alternative']
        self.technologies = self.default_technologies if technologies is None else technologies
        self.component_designs = {} if component_designs is None else component_designs
        if generate_starting_designs and component_designs is None:
            self.make_default_designs(1)
        self.ship_designs = set() if ship_designs is None else set(ship_designs)
        self.__visible_bodies = [] if visible_bodies is None else visible_bodies
        self.__visible_bodies_updated = -1 if visible_bodies_updated is None else visible_bodies_updated
        self.starting_station = None if starting_station is None else starting_station
        self.new_rock(self.spawn_buid)
        self.rock_database[self.spawn_buid]['survey'] = True

    def export_encode(self):
        e = {**self.__dict__}
        e['ships'] = list(e['ships'])
        return e

    @classmethod
    def import_decode(cls, uni, d):
        imp = cls(
            uni=uni,
            seed=Seed(*d['seed']),
            name=d['name'],
            pid=d['pid'],
            ships=set(d['ships']),
            technologies=d['technologies'],
            designs=d['designs'],
            cuids=d['cuids'],
            visible_bodies=d['_Player__visible_bodies'],
            visible_bodies_updated=d['_Player__visible_bodies_updated'],
            starting_station=d['starting_station'],
            )
        return imp

    def alert(self, message, halt=False, debug=True):
        self.notifications.insert(0, message)
        self.notifications = self.notifications[:50]
        if debug:
            self.uni.debug(message, force_print=True)
        if halt:
            self.uni.request_simulation_halt()

    # GENESIS
    @property
    def default_technologies(self):
        # t = {ctype: {tname: self.get_starting_tech_seed() for tname in self.__default_technology_names} for ctype in common.CTYPES}
        t = {}
        return t

    def make_default_designs(self, count=1):
        designs = [
            # self.make_component_design(ctype='Engineer', mass=10, material=0, manufacturing=3, technology=3, name=f'Prehistoric Engineer'),
            # self.make_component_design(ctype='Cargo', mass=100, material=0, manufacturing=3, technology=3, name=f'Prehistoric Cargo'),
            # self.make_component_design(ctype='Fuel tank', mass=10, material=0, manufacturing=3, technology=3, name=f'Prehistoric Fuel tank'),
            # self.make_component_design(ctype='Drive', mass=20, material=0, manufacturing=3, technology=3, name=f'Prehistoric Drive'),
            # self.make_component_design(ctype='Miner', mass=100, material=0, manufacturing=3, technology=3, name=f'Prehistoric Miner'),
            # self.make_component_design(ctype='Builder', mass=100, material=0, manufacturing=3, technology=3, name=f'Prehistoric Builder'),
            # self.make_component_design(ctype='Sensor', mass=20, material=0, manufacturing=3, technology=3, name=f'Prehistoric Sensor'),
            ]

        for d in designs:
            self.add_component_design(d.stats)

    def generate_starting_ships(self, cuids):
        for i, ship_cuid in enumerate(cuids):
            sc = self.uni.subclasses[ship_cuid]
            self.uni.debug(f'Generating starting ship: {adis(sc)}')
            ship_name = sc["name"]
            ship_cargo = {}
            # First starting ship (main station) gets different treatment
            if i == 0:
                ship_name = f'{self.uni.subclasses[ship_cuid]["name"]}'
                ship_cargo = {common.E.NAMES[_]: int(STARTING_ARADIUM/(_+1)) for _ in (0, 2, 4, 6)}
            ship_buid = self.uni.new_ship(
                name=ship_name,
                subclass=ship_cuid,
                starting_position=self.uni.moria,
                starting_velocity=np.zeros(2),
                allegiance=self.pid)
            self.new_ship(ship_buid)
            sb = self.uni.id2body(ship_buid)
            if common.cargo_mass(ship_cargo) > 0:
                sb.do_cargo_add(ship_cargo)
                sb.fuel_tank = sb.stats['fuel_tank']

    def build_new_ship(self, name, subclass):
        self.new_ship(self.uni.new_ship(
            name=name, subclass=subclass,
            starting_position=self.uni.moria,
            allegiance=self.pid))

    # SHIPS
    def new_ship(self, buid):
        assert buid not in self.ships
        if len(self.ships) == 0:
            self.starting_station = buid
        self.ships.add(buid)

    def remove_ship(self, buid):
        if buid not in self.ships:
            self.alert(f'Trying to remove ship {buid} but not found in player ships: {self.ships}.')
            return
        self.ships.discard(buid)
        if len(self.ships) == 0:
            self.alert('NO SHIPS LEFT - GAME OVER :(')
            self.uni.game_over()

    # SHIP DESIGNS
    def add_ship_class(self, stats):
        name = stats['metadata']['name']
        if name == '':
            self.alert(f'Ship class must have a name.')
            return
        # Add to universe and to our cuids
        new_cuid = self.uni.add_subclass(
            name, stats['metadata']['description'], stats)
        self.ship_designs.add(new_cuid)

        self.alert(f'Received ship class design, new ship class available: {name}.')
        self.uni.debug(f'Based on stats: {adis(stats)}')

    def make_ship_design(self, ship_design_spec):
        # Eventually this will modify the parameters or the end result based on player-based modifers.
        comps = []
        for duid, count in ship_design_spec.items():
            for _ in range(count):
                comps.append(self.component_designs[duid]['stats'])
        # TODO add player stuff to metadata
        return components.ShipDesign(comps)

    # COMPONENT DESIGNS
    def make_component_design(self, ctype, *args, **kwargs):
        # Eventually this will modify the parameters or the end result based on player-based modifers.
        return components.COMP_TYPES[ctype](*args, **kwargs)

    def add_component_design(self, stats):
        duid = len(self.component_designs)
        self.component_designs[duid] = {
            'obsolete': False,
            'stats': stats,
            }
        self.alert(f'Made new {stats["metadata"]["ctype"]} design: {stats["metadata"]["name"]}')
        self.uni.debug(f'New DUID stats: {adis(stats)}')

    def make_design_obsolete(self, duid):
        self.component_designs[duid]['obsolete'] = True

    # TECH
    def get_starting_tech_seed(self):
        t = common.Tech(self.seed.r)
        while t.normal_value < 0.45 or t.normal_value > 0.55:
            t = common.Tech(self.seed.r)
        return t.s.seed

    # UNIVERSE INTERFACE
    def new_rock(self, rock_buid):
        if rock_buid in self.rock_database:
            return
        self.rock_database[rock_buid] = {
            'survey': False,
            }

    # CLASS PROPERTIES
    def is_visible(self, buid):
        return buid in self.visible_bodies

    @property
    def visible_buids(self):
        if self.uni.time != self.__visible_bodies_updated:
            new_set = set()
            for ship in self.ships:
                new_set.add(ship)
                new_set.update(self.uni.id2body(ship).visible_bodies)
            self.__visible_bodies = list(new_set)
            self.__visible_bodies_updated = self.uni.time
        return self.__visible_bodies

    def get_visible_bodies(self, refresh=True):
        if refresh:
            return self.visible_bodies
        return self.__visible_bodies

    @property
    def idle_ships(self):
        idles = set()
        for ship_buid in self.ships:
            ship = self.uni.id2body(ship_buid)
            if ship.bridge.is_idle:
                idles.add(ship_buid)
        return idles

    @property
    def visible_ships(self):
        return [*filter(self.visible_bodies, lambda x: self.uni.id2body(x).is_ship)]

    @property
    def total_repair_costs(self):
        costs = {}
        annum_costs = {}
        for s in self.ships:
            scost = self.uni.id2body(s).repair_cost()
            DictOp.add(costs, scost)
        return costs

    @property
    def total_maintenance_costs(self):
        costs = {}
        for s in self.ships:
            DictOp.add(costs, self.uni.id2body(s).stats['maint_cost'])
        return costs


# SHIP BRIDGE

class Bridge:
    """
    The bridge represents the decision making component of a ship. It will take orders and attempt to dutifully perform them.
    """
    def __init__(self, ship, tags=None,
                 current_order=None, order_queue=None):
        self.ship = ship
        self.__status = 'Nominal.'
        self.__pending_order = ''
        self.tags = list() if tags is None else tags
        self.current_order_id = ''
        self.current_order = None
        if current_order is not None:
            raise NotImplementedError(f'Bridge current_order kwarg not implemented yet.')
        else:
            self.set_idle()
        self.order_queue = [] if order_queue is None else order_queue

    def set_tags(self, tags=None):
        if isinstance(tags, str):
            tags = [tags]
        if tags is None:
            self.tags = list()
        else:
            for tag in tags:
                if tag != '' and tag not in self.tags:
                    self.tags.append(tag)

    @property
    def status(self):
        return self.__status

    def set_idle(self, summary=None):
        p = {}
        if summary:
            p = {'summary': str(summary)}
        self.current_order = Idle(ship=self.ship, params=p)

    @property
    def uni(self):
        return self.ship.uni

    def initialize(self):
        # self.d['']
        pass

    def update_order(self, params):
        self.current_order.update_params(params)

    def give_orders(self, orders, queue=False):
        assert is_iterable(orders)
        for otype, params in orders:
            assert otype in ORDER_TYPES
            assert isinstance(params, dict)
        if not queue:
            self.order_queue = orders
        else:
            self.order_queue.extend(orders)
        self.ship.player.alert(f'{self.ship.fname} {"queued" if queue else "took"} {len(orders)} order{"s" if len(orders) > 1 else ""}.')
        self.uni.debug(adis(orders))
        if not queue:
            self.__status = f'Preparing for {orders[0][0]} order.'
            self.__pending_order = str(Seed().r)+h256(self.order_queue)
            self.add_timed_callback(
                self.uni.time+self.uni.PLANK,
                'do_next_order',
                {'__pending_order__': self.__pending_order})

    def do_next_order(self, context):
        if context['__pending_order__'] != self.__pending_order:
            # Prune obsolete orders
            return
        self.uni.debug(f'{self.ship.fname} order queue:\n{adis(self.order_queue)}')
        if len(self.order_queue) == 0:
            # self.current_order = Idle(ship=self.ship, params={})
            self.set_idle('Completed all orders.')
            self.ship.request_guidance(f'completed all orders.')
            # self.__current_state = 'Complete'
            return
        self.__status = 'Nominal.'
        next_order_type, next_order_params = self.order_queue.pop(0)
        self.set_current_order(next_order_type, next_order_params)

    def set_current_order(self, otype, params):
        self.uni.debug(f'{self.ship.name} bridge next order: {otype}\n{params}')
        self.current_order_id = h256(f'{self.uni.time}{otype}{params}')
        self.current_order = ORDER_TYPES[otype](ship=self.ship, params=params)
        self.current_order.start_order()

    def order_complete(self):
        self.__pending_order = h256(self.order_queue)
        self.do_next_order({'__pending_order__': self.__pending_order})

    def add_timed_callback(self, time, callback, context):
        self.uni.add_timed_callback(
            time, 'bridge', self.ship.buid,
            callback, context)

    def add_order_timed_callback(self, time, callback, context):
        self.uni.add_timed_callback(
            time, 'bridge', self.ship.buid,
            'order_callback',
            {
                **context,
                '__bridge_order_id__': self.current_order_id,
                '__bridge_order_callback__': callback,
            })

    def order_callback(self, context):
        order_id = context['__bridge_order_id__']
        if order_id != self.current_order_id:
            return
        callback = context['__bridge_order_callback__']
        f = getattr(self.current_order, callback)
        f(context)

    @property
    def is_idle(self):
        return isinstance(self.current_order, Idle)

    def tags_formatted(self, max_tags=2):
        if len(self.tags) == 0:
            return 'No tags.'
        elif len(self.tags) > max_tags:
            return f'{", ".join(self.tags[:max_tags-1])} (+{len(self.tags)-max_tags+1})'
        else:
            return ', '.join(self.tags)


class Order:
    """
    An order represents a single order performed by the bridge. Each order subclass takes a parameter dict and handles its own state and logic. Generally a ship will only have a single Order class instance at a time - this gives it full control of the ship.
    """
    # Each order's PARAMS are used for reference by the UI to determine what parameters an order expects. Key/value format is as follows:
    # 'parameter_internal_name': ('parameter_type', 'default_value', 'human_readable_label')
    def __init__(self, ship, params, dcache=None):
        self.ship = ship
        self.params = params
        self.dcache = {} if dcache is None else dcache
        self._debug = False if 'debug' not in params else params['debug']

    @classmethod
    def resolve_param_element(cls, ship, param):
        assert is_iterable(param)
        for element in param:
            assert element in common.E.NAMES
        return param

    @classmethod
    def resolve_param_float(cls, ship, param):
        return float(param)

    @classmethod
    def resolve_param_position(cls, ship, param):
        if isiterable(param):
            if not isinstance(param, str) and len(param) == 2:
                return np.array(param)
        if isinstance(param, str):
            if param in ship.uni.bodies:
                return ship.uni.id2body(param).position
            return next(ship.uni.gen_bodies(names=[param])).position
        raise NotImplementedError(f'position param needs a coordinate or ship name')

    @classmethod
    def resolve_param_ship(cls, ship, param):
        assert is_iterable(param)
        for buid in param:
            assert buid in ship.uni.bodies
        return param

    @classmethod
    def resolve_param_ship_class(cls, ship, param):
        assert is_iterable(param)
        for cuid in param:
            assert cuid in ship.player.ship_designs
        return param

    @classmethod
    def resolve_param_rock_class(cls, ship, param):
        assert is_iterable(param)
        for cuid in param:
            assert cuid in ship.uni.rock_subclasses
        return param

    @classmethod
    def resolve_param_cargo(cls, ship, param):
        assert isinstance(param, dict)
        for ename, count in param.items():
            assert ename in common.E.NAMES
            assert isinstance(count, int) or isinstance(count, float)
        return param

    @classmethod
    def resolve_param_tags(cls, ship, param):
        assert isiterable(param, param)
        return param

    @classmethod
    def resolve_summary(cls, params):
        raise RuntimeError(f'Calling resolve_summary of Order base class, must subclass to use.')

    def update_params(self, params):
        assert isinstance(params, dict)
        for k, v in params.items():
            if k not in self.params:
                print(f'{self} found unknown param for update: {k}. Aborted updating params.')
                return
        self.uni.debug(f'Updating params for {self}.\n{adis(params)}')
        for k, v in params.items():
            self.params[k] = v

    def drive(self, target, speed, breaks=True):
        return self.ship.order_drive(target, speed, breaks)

    def mine(self, elements):
        target_buid = None
        # Cycle through elements to find best available
        for target_element in elements:
            # Find best target for element, otherwise, try next element
            for target_buid in self.ship.proximal(ships=False):
                if self.ship.check_order_mine(target_buid, target_element):
                    break
            else:
                continue
            # If target found, we have element and target rock
            break
        else:
            # If no target for no element found, return None indicating that nothing was performed
            return None
        # Order the ship to mine
        mine_complete = self.ship.order_mine(target_buid, target_element)
        # Return when complete to order
        return mine_complete

    def build(self, cuids):
        target_cuid = None
        # Cycle through ship classes until we find one we can build
        for target_cuid in cuids:
            if not self.ship.build_check(target_cuid):
                continue
            break
        else:
            # If no ship class was found, return None indicating that nothing was performed
            return None
        # Return when order completes
        build_complete = self.ship.order_build(target_cuid)
        return build_complete

    def cargo_set(self, target_buid, item, count):
        assert count >= 0
        diff = count - self.ship.cargo_hold[item]
        if diff == 0:
            return None
        take = diff > 0
        self.ship.order_transfer_cargo(
            target_buid=target_buid, item=item,
            count=math.fabs(diff), take=take)
        return None

    def repair(self, target_buid, percent):
        if self._debug:
            self.uni.debug(f'{self.ship.fname} trying to repair from: {target_buid}')
        self.uni.id2body(target_buid).command_serve(self.ship.buid, percent)

    def serve(self, target_buid, percent):
        if self._debug:
            self.uni.debug(f'{self.ship.fname} trying to serve: {target_buid}')
        self.ship.command_serve(target_buid, percent)

    def cargo_fill(self, items, **kwargs):
        for ship_buid in self.ship.proximal(allegiance=self.ship.allegiance, rocks=False, **kwargs):
            tb = self.uni.id2body(ship_buid)
            if self.ship.remaining_hold == 0:
                break
            for item in items:
                if self.ship.remaining_hold == 0:
                    break
                if tb.cargo_hold[item] > 0:
                    self.cargo_set(ship_buid, item, float('inf'))

    def cargo_dump(self, items, **kwargs):
        for ship_buid in self.ship.proximal(allegiance=self.ship.allegiance, rocks=False, **kwargs):
            tb = self.uni.id2body(ship_buid)
            if self.ship.cargo_mass == 0:
                break
            for item in items:
                if tb.remaining_hold == 0:
                    break
                if self.ship.cargo_hold[item] > 0:
                    self.cargo_set(ship_buid, item, 0)

    def find_repair(self, include_self=True, **kwargs):
        for ship_buid in self.ship.proximal(allegiance=self.ship.allegiance, rocks=False, include_self=include_self, **kwargs):
            if self.ship.maintenance_condition == 1:
                break
            self.repair(ship_buid, 1)

    def find_serve(self, include_self=True, **kwargs):
        for ship_buid in self.ship.proximal(allegiance=self.ship.allegiance, rocks=False, include_self=include_self, **kwargs):
            self.serve(ship_buid, 1)

    def find_refuel(self, include_self=True, **kwargs):
        for target_buid in self.ship.proximal(allegiance=self.ship.allegiance, rocks=False, include_self=include_self, **kwargs):
            if self.ship.fuel_tank == self.ship.stats['fuel_tank']:
                break
            self.ship.order_refuel(target_buid=target_buid)

    def find_birth(self, buids=None, **kwargs):
        for ship_buid in self.uni.filter_bodies(
            bodies=self.ship.proximal(allegiance=self.ship.allegiance, rocks=False, **kwargs) if buids is None else buids):
            tb = self.uni.id2body(ship_buid)
            if self._debug:
                self.uni.debug(f'{self.ship.fname} trying to birth: {tb.fname}')
            tb.command_dock(self.ship.buid)

    def find_unbirth(self, buids=None, **kwargs):
        for ship_buid in self.uni.filter_bodies(
            bodies=self.ship._tugging if buids is None else buids,
            allegiance=self.ship.allegiance,
            rocks=False, **kwargs):
            tb = self.uni.id2body(ship_buid)
            if self._debug:
                self.uni.debug(f'{self.ship.fname} unbirthing: {tb.fname}')
            tb.command_undock()

    @property
    def uni(self):
        return self.ship.uni

    def start_order(self):
        raise RuntimeError(f'Calling start_order of Order base class, must subclass to use.')

    def end_order(self, context=None):
        self.ship.player.alert(f'{self.ship.name} completed delivery order.')
        self.ship.bridge.order_complete()

    @property
    def summary(self):
        return self.__class__.resolve_summary(self.params)

    @property
    def current_state(self):
        return 'Nominal.'

    @property
    def add_timed_callback(self):
        return self.ship.bridge.add_order_timed_callback

    @property
    def bridge(self):
        return self.ship.bridge


class Idle(Order):
    PARAMS = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def resolve_summary(cls, params):
        return 'Idle.' if 'summary' not in params else params['summary']

    def start_order(self):
        self.uni.debug(f'{self.ship.name} going idle.')


class Station(Order):
    PARAMS = {
        'repair': ('bool', True, 'Repair station'),
        'serve': ('bool', False, 'Service guests'),
        'mining_element': ('element', common.E.SR, 'Mine element'),
        'cargo_dump': ('slist', '', 'Offload to tags'),
        'building_class': ('slist', '', 'Build ship'),
        # 'max_build_ships': ('float', 0, 'Max ships to build'),
        }
    INTERVAL = 100
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dcache['mining'] = -1
        self.dcache['building'] = -1

    @property
    def building_cuids(self):
        return Order.resolve_param_ship_class(self.ship, self.params['building_class'])

    @property
    def mining_elements(self):
        return Order.resolve_param_element(self.ship, self.params['mining_element'])

    @classmethod
    def resolve_summary(cls, params):
        mine_str = ''
        build_str = ''
        cargo_str = ''
        if 'mining_element' in params:
            s = sjoin([e[0] for e in params["mining_element"]], split='')
            mine_str = f' | Mining: {adis(s, max_length=7)}'
        if 'building_class' in params:
            if len(params["building_class"]) > 0:
                if len(params["building_class"]) > 1:
                    s = params["building_class"][0]
                elif len(params["building_class"]) > 1:
                    s = f'{len(params["building_class"])} classes'
                build_str = f' | Building: {adis(s, 15)}'
        # if 'cargo_reserve' in params:
        #     cargo_str = f' | Reserving: {diss(params["cargo_reserve"], 20)}'
        r = f'Standing station{mine_str}{build_str}'  #'{cargo_str}'
        return r

    def start_order(self):
        self.bridge.set_tags('Station')
        self.interval_check()

    def interval_check(self, context=None):
        if self.params['debug']:
            self.uni.debug(f'{self.ship.fname} station interval_check')
        self.check_mine()
        self.check_build()
        if self.params['repair']:
            self.find_repair()
        if self.params['serve']:
            self.find_serve(include_self=False)
        self.add_timed_callback(
            self.uni.time+Station.INTERVAL,
            'interval_check',
            {'summary': f'{self.ship.fname} station order next interval'})

    def check_mine(self, context=None):
        if self._debug:
            self.uni.debug(f'{self.ship.fname} station check_mine')
        if self.params['cargo_dump']:
            tags = Order.resolve_param_tags(self, self.params['cargo_dump'])
            self.cargo_dump(tags=tags)
        if self.params['debug']:
            self.uni.debug(f'{self.ship.fname} station check_mine')
        if self.dcache['mining'] < self.uni.time:
            next_mining = self.mine(self.mining_elements)
            if next_mining is not None:
                self.dcache['mining'] = next_mining
                self.add_timed_callback(next_mining+common.PLANK, 'check_mine', {'summary': f'{self.ship.fname} station order next mine'})

    def check_build(self, context=None):
        if self._debug:
            self.uni.debug(f'{self.ship.fname} station check_build')
        if self.dcache['building']  < self.uni.time:
            next_building = self.build(self.building_cuids)
            if self._debug:
                self.uni.debug(f'{self.ship.fname} station next_building: {next_building}')
            if next_building is not None:
                self.dcache['building'] = next_building
                self.add_timed_callback(
                    next_building+common.PLANK,
                    'check_build',
                    {'summary': f'{self.ship.fname} station order next build'})

    @property
    def current_state(self):
        s = []
        if self.dcache["mining"] > self.uni.time:
            s.append(f'Mining: {round(self.dcache["mining"]-self.uni.time, 1)}')
        if self.dcache["building"] > self.uni.time:
            s.append(f'Building: {round(self.dcache["building"]-self.uni.time, 1)}')
        if len(s) > 0:
            return sjoin(s, split=' | ')
        return f'Standing.'


class Deliver(Order):
    PARAMS = {
        'source': ('position', '', 'From'),
        'destination': ('position', '', 'To'),
        'speed': ('float', '', 'Speed'),
        'repair_source': ('bool', True, 'Repair at source'),
        'refuel_source': ('bool', True, 'Refuel at source'),
        'service_source': ('bool', False, 'Service at source'),
        'source_unload_tags': ('slist', '', 'Unload to'),
        'haul_cargo': ('element', '', 'Haul cargo'),
        'source_load_tags': ('slist', '', 'Load from'),
        'tug_ships': ('slist', '', 'Tug ships'),
        'repair_destination': ('bool', False, 'Repair at destination'),
        'refuel_destination': ('bool', False, 'Refuel at destination'),
        'service_destination': ('bool', False, 'Service at destination'),
        'return': ('bool', True, 'Return'),
        'dest_unload_tags': ('slist', '', 'Unload to'),
        'haul_cargo_back': ('element', '', 'Haul cargo back'),
        'dest_load_tags': ('slist', '', 'Load from'),
        'tug_ships_back': ('slist', '', 'Tug ships back'),
        'repeat': ('bool', False, 'Repeat'),
        'min_interval': ('float', 100, 'Minimum round time'),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__current_state = 'Starting delivery.'
        self.dcache['last_depart'] = -float('inf')
        if self.params['source'] == '':
            self.params['source'] = self.ship.position

    @classmethod
    def resolve_summary(cls, params):
        r = f'Delivery'
        return r

    @property
    def tugging_buids(self):
        return Order.resolve_param_ship(self.ship, self.params['tug_ships'])

    @property
    def tugging_buids_back(self):
        return Order.resolve_param_ship(self.ship, self.params['tug_ships_back'])

    def start_order(self):
        self.bridge.set_tags('Deliver')
        source = Order.resolve_param_position(self, self.params['source'])
        dest = Order.resolve_param_position(self, self.params['destination'])
        if vmag(self.ship.position - dest) <= vmag(self.ship.position - source):
            self.go_destination(context={'start': True})
        else:
            self.go_source(context={'start': True})

    def go_destination(self, context=None):
        min_interval = max(1, Order.resolve_param_float(self, self.params['min_interval']))
        interval_ready = self.dcache['last_depart'] + min_interval
        if self.uni.time < interval_ready:
            self.__current_state = f'Resting before next round ({interval_ready}).'
            self.add_timed_callback(interval_ready, 'go_destination', context={'summary': 'Resting before next round'})
            return
        self.__current_state = 'Going to destination.'
        dest = Order.resolve_param_position(self, self.params['destination'])
        speed = max(0.1, Order.resolve_param_float(self, self.params['speed']))
        if self._debug:
            self.uni.debug(f'{self.ship.fname} going to destination: {dest} @{speed}')
        drive_complete = self.drive(dest, speed)
        if drive_complete > 0:
            self.__current_state = 'Driving to destination.'
            self.add_timed_callback(
                drive_complete, 'do_destination', context={
                    'summary': 'Arrive at deliver destination',
                    'depart_time': self.uni.time,
                    })
        else:
            self.do_destination(context={**context, 'depart_time': self.uni.time})

    def do_destination(self, context=None):
        if context is None:
            context = {}
        self.__current_state = 'Delivering at destination.'
        self.dcache['last_depart'] = context['depart_time']
        dest = Order.resolve_param_position(self, self.params['destination'])
        if not self.ship.at_target(dest):
            self.ship.request_guidance(f'failed to arrive at destination: {dest}.')
            self.__current_state = 'Failed to arrive at destination'
            return
        if self._debug:
            self.uni.debug(f'{self.ship.fname} do_destination')

        # Refuel, repair, service, dump cargo, and unbrith
        if self.params['refuel_destination']:
            self.find_refuel()
        if self.params['repair_destination']:
            self.find_repair()
        if self.params['service_destination']:
            self.find_serve()
        self.cargo_dump(
            items=Order.resolve_param_element(self, self.params['haul_cargo']),
            tags=Order.resolve_param_tags(self, self.params['dest_unload_tags']))
        self.find_unbirth(self.tugging_buids)

        if self.params['return'] or 'start' in context:
            self.cargo_fill(
                items=Order.resolve_param_element(self, self.params['haul_cargo_back']),
                tags=Order.resolve_param_tags(self, self.params['dest_load_tags']))
            self.find_birth(self.tugging_buids_back)
            self.go_source()
        else:
            self.__current_state = 'Completed order.'
            self.end_order()

    def go_source(self, context=None):
        self.__current_state = 'Returning to source.'
        source = Order.resolve_param_position(self, self.params['source'])
        speed = Order.resolve_param_float(self, self.params['speed'])
        if self._debug:
            self.uni.debug(f'{self.ship.fname} going to source: {source} @{speed}')
        drive_complete = self.drive(source, speed)
        if drive_complete > 0:
            self.__current_state = 'Driving to source.'
            self.add_timed_callback(drive_complete, 'do_source', context={'summary': 'Arrive at deliver source'})
        else:
            self.do_source(context=context)

    def do_source(self, context=None):
        if context is None:
            context = {}
        self.__current_state = 'Delivering at source.'
        source = Order.resolve_param_position(self, self.params['source'])
        if not self.ship.at_target(source):
            self.ship.request_guidance(f'failed to arrive at source: {source}.')
            self.__current_state = 'Failed to arrive at source'
            return
        if self._debug:
            self.uni.debug(f'{self.ship.fname} do_source')
        # Refuel, repair, service, dump cargo, and unbrith
        if self.params['refuel_source']:
            self.find_refuel()
        if self.params['repair_source']:
            self.find_repair()
        if self.params['service_source']:
            self.find_serve()
        self.cargo_dump(
            items=Order.resolve_param_element(self, self.params['haul_cargo_back']),
            tags=Order.resolve_param_tags(self, self.params['source_unload_tags']))
        self.find_unbirth(self.tugging_buids_back)

        if self.params['repeat'] or 'start' in context:
            self.cargo_fill(
                items=Order.resolve_param_element(self, self.params['haul_cargo']),
                tags=Order.resolve_param_tags(self, self.params['source_load_tags']))
            self.find_birth(self.tugging_buids)
            self.go_destination()
        else:
            self.__current_state = 'Completed order.'
            self.end_order()

    @property
    def current_state(self):
        return self.__current_state


class Explore(Order):
    """Explore and survey rocks. Will refuel at base, then go to nearest rock and survey. May continue going to next nearest rock until out of fuel. Return to base and may repeat."""

    PARAMS = {
        'base': ('position', '', 'Refuel Base'),
        'repair_base': ('bool', True, 'Repair at base'),
        'destination': ('position', '', 'Explore'),
        'speed': ('float', '', 'Speed'),
        'survey': ('bool', True, 'Survey'),
        'auto_next': ('bool', True, 'Auto explore next'),
        'skip_surveyed': ('bool', True, 'Skip surveyed'),
        'smart_path': ('bool', True, '\"Smart\" pathfinding'),
        'start_base': ('bool', True, 'Start at base'),
        'explore_cuids': ('slist', '', 'Explore classes'),
        'survey_cuids': ('slist', '', 'Survey classes'),
        'min_cond': ('float', 0.5, 'Min condition'),
        'repeat': ('bool', True, 'Repeat'),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__current_state = 'Starting exploration.'
        self.dcache['explored'] = []

    @classmethod
    def resolve_summary(cls, params):
        r = f'Exploring'
        return r

    def start_order(self):
        self.bridge.set_tags('Explore')
        if self.params['start_base']:
            self.go_base(context={'start': True})
        else:
            self.go_next(context={'start': True})

    def go_base(self, context=None):
        self.__current_state = 'Returning to base.'
        if self.params['base'] == '':
            if 'start' in context:
                self.go_next(context=context)
            else:
                self.__current_state = 'Completed order.'
                self.end_order()
            return
        base = self.resolve_param_position(self, self.params['base'])
        speed = self.resolve_param_float(self, self.params['speed'])
        if self._debug:
            self.uni.debug(f'{self.ship.fname} going to base: {base} @{speed}')
        drive_complete = self.drive(base, speed)
        if drive_complete > 0:
            self.__current_state = 'Driving to base.'
            self.add_timed_callback(drive_complete, 'do_base', context={'summary': 'Arrive at explore base'})
        else:
            self.do_base(context=context)

    def do_base(self, context=None):
        if context is None:
            context = {}
        self.__current_state = 'Resupplying at base.'
        base = Order.resolve_param_position(self, self.params['base'])
        if not self.ship.at_target(base):
            self.ship.request_guidance(f'failed to arrive at base: {base}.')
            self.__current_state = 'Failed to arrive at base'
            return
        if self._debug:
            self.uni.debug(f'{self.ship.fname} do_base')
        # Refuel
        self.find_refuel()
        if self.params['repair_base']:
            self.find_repair()

        if self.params['repeat'] or 'start' in context:
            self.go_next(context)
        else:
            self.__current_state = 'Completed order.'
            self.end_order()

    def go_next(self, context=None):
        self.__current_state = 'Exploring next object.'
        speed = max(0.1, Order.resolve_param_float(self, self.params['speed']))
        if ('next' in context and self.params['smart_path']) or self.params['destination'] == '':
            dest = self.ship.position
        else:
            dest = Order.resolve_param_position(self, self.params['destination'])
        if self._debug:
            self.uni.debug(f'{self.ship.fname} finding next rock to explore. Already explored: {self.dcache["explored"]}')
        # Find next destination
        rocks = self.ship.player.rock_database.keys()
        potential_rocks = list(sorted(
            rocks, key=lambda x, t=dest: self.uni.sort_by_distance_key(x, t)))
        explore_cuids = Order.resolve_param_rock_class(self, self.params['explore_cuids'])
        if len(potential_rocks) == 0:
            self.ship.request_guidance(f'{self.name} found nothing to explore.')
            self.__current_state = 'Nothing to explore'
            return
        for rock_buid in potential_rocks:
            if rock_buid in self.dcache['explored']:
                continue
            if self.params['skip_surveyed']:
                if self.ship.player.rock_database[rock_buid]['survey']:
                    continue
            rock = self.uni.id2body(rock_buid)
            if len(explore_cuids) > 0:
                if rock.subclass not in explore_cuids:
                    continue
            next_rock = rock
            break

        # Check for fuel and return to base if needed
        if not self.ship.can_reach(target=next_rock.position,
                                   speed=speed,
                                   include_return=self.params['repeat']):
            # If we just started the order and lacking fuel before our first explore target, request guidance
            if 'start' in context:
                self.ship.request_guidance(f'Lacking fuel for requested speed.')
                self.__current_state = 'Lacking fuel'
                return
            self.go_base(context={'repeat': True})
            return
        if self.ship.maintenance_condition < self.params['min_cond'] and self.params['repeat']:
            self.go_base(context={'repeat': True})
            return

        if self._debug:
            self.uni.debug(f'{self.ship.fname} found next object: {next_rock.fname} (going @{speed})')

        drive_complete = self.drive(next_rock.position, speed)
        if drive_complete > 0:
            self.__current_state = 'Driving to destination.'
            self.add_timed_callback(
                drive_complete, 'do_destination',
                context={
                    'summary': f'Arrive at explore destination {next_rock.fname}',
                    'rock_buid': rock_buid,
                    })
        else:
            self.do_destination(context={'rock_buid': rock_buid})

    def do_destination(self, context):
        if self._debug:
            self.uni.debug(f'{self.ship.fname} do_destination')
        rock_buid = context['rock_buid']
        rock = self.uni.id2body(rock_buid)
        self.__current_state = f'Surveying {rock.name}.'
        self.dcache['explored'].append(rock_buid)
        # Survey
        if self.params['survey']:
            survey_cuids = Order.resolve_param_rock_class(self, self.params['survey_cuids'])
            # if self._debug:
            #     self.uni.debug(f'rock cuid: {rock.subclass} | survey cuids: {survey_cuids} (from {self.params["explore"]})')
            if len(survey_cuids) == 0 or rock.subclass in survey_cuids:
                survey_complete = self.ship.order_survey(rock_buid)
                if survey_complete > 0:
                    if self._debug:
                        self.uni.debug(f'{self.ship.fname} surveying {rock.fname}.')
                    self.__current_state = f'Surveying {rock.name}: {survey_complete}'
                    if self.params['auto_next']:
                        self.add_timed_callback(
                            survey_complete, 'go_next',
                            context={'summary': f'Surveying {rock_buid}.', 'next': True})
                    else:
                        self.add_timed_callback(
                            survey_complete, 'end_order',
                            context={'summary': f'Surveying {rock_buid}.'})
                    return
        if self.params['auto_next']:
            self.go_next(context={'next': True})
        else:
            self.add_timed_callback(
                survey_complete, 'end_order',
                context={'summary': f'Surveying {rock_buid}.'})


    @property
    def current_state(self):
        return self.__current_state



ORDER_TYPES = {
    'Idle': Idle,
    'Station': Station,
    'Deliver': Deliver,
    'Explore': Explore,
    }


def vmag(a):
    return np.linalg.norm(a)
