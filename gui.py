# GUI - Kivy based

from nutil import *
from nutil import kex
from nutil import lists
from nutil.kex import kv
from nutil.kex import BaseWidgets as W
from nutil.kex import xwidgets as XW

import time
import numpy as np

import universe
import common
import agency
import components


DEFAULT_CONFIG = {
    'fps': 30,
    'simulate_interval': 1,
    'bgcolor': (Seed().r, 1, 0.1),
    # Font and font size only matter for specific widgets not extended by kxw. kxw should probably be deprecated for being able to set default font.
    'font': 'assets/fonts/ubuntu/UbuntuMono-R.ttf',
    'font_size': 18,
    }


class App(XW.App):
    def __init__(self, uni=None, **kwargs):
        print(f'Starting App - {universe.FULL_TITLE}...')
        super().__init__(**kwargs)
        self.auto_play = -1
        self.__last_auto_play = ping()
        self.root.make_bg(DEFAULT_CONFIG['bgcolor'])
        self.title = universe.FULL_TITLE
        self.icon = 'icon.png'
        kex.set_win_size((1700, 1000))
        self.__temp_loading_frame = self.root.add(XW.Box(orientation='vertical'))
        self.__temp_loading_frame.add(W.Label(text='Loading...'))
        self.__temp_loading_progress = self.__temp_loading_frame.add(W.ProgressBar())
        # We schedule the rest to happen after the next frame, to allow the
        #    title and icon to refresh before starting the time consuming
        #    process of generating a universe and full GUI.
        kv.Clock.schedule_once(lambda dt, u=uni, *a: self.make_widgets(u), 0.1)

    def set_autoplay(self, value=None):
        if value is None:
            value = -self.auto_play
        self.auto_play = value
        print('Autoplay set to:', self.auto_play)
        self.__last_auto_play = ping()

    @property
    def dev_mode(self):
        # return self.__dev_mode
        return self.ucache.ref('dev_mode')

    # SECONDARY INIT METHODS
    def make_widgets(self, uni):
        # UNIVERSE
        print('Creating universe...')
        self.uni = universe.Universe() if uni is None else uni
        self.selected_target = self.uni.moria

        # GUI
        print('Building GUI...')
        self.ucache = TrackCache()
        self.register_ucache()
        self.root.orientation = 'vertical'

        # MAIN FRAME
        self.main_frame = Main()
        self.main_view_switching = {sname: lambda *a, s=sname: self.main_frame.switch_view(s) \
                           for sname in self.main_frame.views.keys()}

        # MENU FRAME
        self.menu_frame = XW.Box().set_size(sizey=XW.LINE_DP_STR)
        self.view_spinner = self.menu_frame.add(XW.Spinner(
            entries={f'{sname}': sname for sname in self.main_frame.views.keys()},
            callback=lambda sn, *a: self.main_frame.switch_view(sn)
            )).set_size(sizex='150dp')
        self.menu_frame.add(XW.Menu(buttons={
            # **self.main_view_switching,
            # 'Debug': self.debug_action,
            }))
        self.notification = self.menu_frame.add(
            XW.Label(callback=self.denotify))

        # HOTKEYS
        self.hotkeys.register_dict({
            # SIMULATE
            f'Play / Pause': ('^+ spacebar', lambda: self.set_autoplay()),
            f'Dismiss alerts': ('^ spacebar', self.denotify),
            # MISC
            # f'Force frame': ('^! f', lambda: self.mainloop_hook(0)),
            })

        self.root.remove_widget(self.__temp_loading_frame)
        self.root.add(self.menu_frame)
        self.root.add(self.main_frame)

        # MAINLOOP HOOK
        kv.Clock.schedule_interval(self.mainloop_hook, 1/DEFAULT_CONFIG['fps'])
        print('App initialization complete.')

    def register_ucache(self):
        self.ucache.register_sources({
            'frame': 0,
            'refresh_all': None,
            'dev_mode': None,  # This is managed by DevView
            'simsched': len(self.uni.simulation_scheduler),
            'window_size': lambda: self.window_size,
            'time': lambda: self.uni.time,
            'notifications': lambda: h256(self.uni.player.notifications),
            'player_ships': lambda: h256(self.uni.player.ships),
            'player_component_designs': lambda: len(self.uni.player.component_designs),
            'player_ship_designs': lambda: len(self.uni.player.ship_designs),
            'bodies': lambda: self.uni.body_count,
            'position_anchor': lambda: (0, 0)
            })

    def mainloop_hook(self, dt):
        if self.uni.sim_halted and self.auto_play > 0:
            self.auto_play = -self.auto_play
            print('Paused autoplay.')
        if self.auto_play > 0:
            sblock = DEFAULT_CONFIG['simulate_interval']
            if pong(self.__last_auto_play)/1000 > sblock:
                self.simulate_time(self.auto_play*sblock, set_as_autoplay=False)
                self.__last_auto_play = ping()
        self.ucache['frame'].tvar.flag()
        devs = ' DEV_MODE -' if self.dev_mode else ''
        self.title = f'{self.formatted_time} -{devs} {self.main_frame.current_view} - {universe.FULL_TITLE}'
        self.view_spinner.text = self.main_frame.current_view
        self.notification.text = adis(
            self.uni.player.notifications[0], max_length=150)
        self.ucache.check(debug=self.dev_mode, skip_debug=('frame'))

    def simulate_time(self, elapsed, set_as_autoplay=True):
        if set_as_autoplay:
            self.auto_play = -elapsed
        self.uni.simulate_time(elapsed)

    @property
    def window_size(self):
        return tuple(kv.Window.size)

    @property
    def visible_buids(self):
        if self.ucache['show_all_bodies'].tvar.ref:
            return self.uni.buids
        return self.uni.player.visible_buids

    def get_visible_buids(self, sort_mass=True):
        if sort_mass:
            return sorted(self.visible_buids, key=lambda x: -self.uni.id2body(x).dry_mass)
        return self.visible_buids

    @property
    def formatted_time(self):
        return format_time(self.uni.time)

    def denotify(self, *args):
        self.menu_frame.make_bg((0, 0, 0, 0))

    @property
    def player(self):
        return self.uni.player


class AppLink:
    def __getattr__(self, a):
        if a in ['uni', 'ucache', 'hotkeys', 'dev_mode', 'player']:
            return getattr(self.app, a)


class Main(XW.Box, AppLink):
    def __init__(self, **kwargs):
        super().__init__(wname='Main', **kwargs)
        print(make_title('Main widget build'))

        # Screen - view manager
        self.view_manager = self.add(XW.ScreenSwitch(
            wname='Main ScreenSwitch',
            transition=kv.FadeTransition(duration=0.05),
            ))
        self.views = {
            'Command': PlayerView(wname='Command view'),
            'Map': MapView(wname='Map view'),
            'Objects': BodyView(wname='Objects view'),
            'Orders': OrdersView(wname='Orders view'),
            'Components': ComponentView(wname='Component designer view'),
            'Ship Classes': ShipClassView(wname='Ship class designer view'),
            'Tech': TechView(wname='Tech view'),
            'Dev': DevView(wname='Dev view'),
        }
        #
        for i, (sname, view) in enumerate(self.views.items()):
            self.view_manager.add_screen(sname, view)
            self.app.hotkeys.register(f'Switch to view: {sname}', f' f{i+1}', lambda s=sname: self.switch_view(s))

    def switch_view(self, sname):
        last_screen = self.current_view
        if self.app.dev_mode:
            print(f'Switching view from {last_screen} to {sname}')
        self.view_manager.switch_screen(sname)
        # Force refresh of newly shown screen if it is already showing
        # For now make it auto refresh
        if last_screen == self.current_view or True:
            if hasattr(self.views[sname], 'refresh'):
                self.views[sname].refresh(force=True)

    @property
    def current_view(self):
        return self.view_manager.current_screen.name


class MainView(XW.Box, AppLink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def refresh(self, force=False):
        if not self.is_visible and not force:
            return
        if self.app.dev_mode:
            print(f'Refreshing {self}')


class TechView(MainView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def refresh(self, force=False):
        if not self.is_visible and not force:
            return
        if self.app.dev_mode:
            print(f'Refreshing {self}')


class DevView(MainView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.orientation = 'vertical'
        left_frame = self.add(XW.Box(orientation='vertical')
                              ).set_size(hintx=.3, sizey=f'{XW.LINE_DP*2}dp')
        right_frame = self.add(XW.Box())

        self.dev_mode_cb = left_frame.add(
            XW.CheckButton(text='Dev mode'))
        self.show_all_bodies = left_frame.add(
            XW.CheckButton(text='Show all bodies'))

        self.simsched = right_frame.add(XW.TreeView())

        self.hotkeys.register_dict({
            f'Dev Debug action': ('^+ f9', self.debug_action),
            })

        self.ucache.register('show_all_bodies', lambda: self.show_all_bodies.active)
        self.ucache.register('dev_mode', lambda: self.dev_mode_cb.active)
        self.ucache.register_calls({
            'refresh_all': self.refresh,
            'time': self.refresh,
            'simsched': self.refresh,
            })

    def debug_action(self, *args):
        self.dev_mode_cb.toggle()
        self.hotkeys.set_debug(self.dev_mode)
        print(sjoin([
            make_title('Debug action'),
            f'Args: {adis(args)}',
            make_title(''),
            f'Dev mode set to: {self.dev_mode}',
            make_title(''),
            ]))
        self.uni.player.alert(f'Dev mode enabled.' if self.dev_mode else f'Dev mode disabled.')

    def refresh(self, force=False):
        if not self.is_visible and not force:
            return
        if self.app.dev_mode:
            print(f'Refreshing {self}')

        formatted_simsched = [f'Simsched size: {len(self.app.uni.simulation_scheduler)}']
        for e in self.uni.simulation_scheduler.entries[:10]:
            t = f'{round(e[0], 4)}'
            summary = e[-1]["summary"] if 'summary' in e[-1] else e[-2]
            formatted_simsched.append(f'{t} | {summary}')
        self.simsched.apply(formatted_simsched)


class ComponentView(MainView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.designer_stats = self.add(ComponentDesigner(
            wname='Component designer stats',
            submit_callback=self.submit_component,
            ))

        accordion = self.add(XW.Accordion(wname='component view'))
        self.browser_stats = accordion.add_item(
            ComponentBrowser(wname='Component designer browser stats'),
            title='Existing compare')
        self.custom_stats = accordion.add_item(
            ComponentDesigner(wname='Component designer custom stats'),
            title='Custom compare')
        accordion.open_item(0)

    def submit_component(self, comp):
        if comp:
            if comp.stats['metadata']['name'] == '':
                print('Please enter a name for the component design.')
                return
            self.player.add_component_design(comp.stats)


class ComponentBrowser(XW.Box, AppLink):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.design_selector = self.add(XW.TreeView(
            wname='Component browser design selector',
            callback=self.select_design)).set_size(hinty=0.4)
        self.stats = self.add(ComponentStats(wname='Component browser stats'))
        self.apply_designs()
        self.ucache.register('player_component_designs', callback=self.apply_designs)

    def apply_designs(self):
        self.design_selector.apply({
            f"«{duid:0>4}» {_['stats']['metadata']['name']}": duid for duid, _ in self.player.component_designs.items()
            })

    def select_design(self, duid):
        self.stats.set_stats(self.player.component_designs[duid]['stats'])


class ComponentDesigner(XW.Box, AppLink):
    def __init__(self, submit_callback=None, submit_text=None, **kwargs):
        super().__init__(**kwargs)
        self.__submit_callback = submit_callback
        self.__selected_element = 0
        submit_text = 'Submit' if submit_text is None else submit_text
        self.orientation = 'vertical'

        params_frame = self.add(XW.Box()).set_size(hinty=0.4)

        params = params_frame.add(XW.Box(
            wname='Component designer parameters',
            orientation='vertical',
            )).set_size(hintx=0.3)
        type_box = params.add(XW.Box())
        type_box.add(XW.Label(text='Type'))
        self.ctype = type_box.add(XW.Spinner(
            entries=components.COMP_TYPES.keys(),
            callback=self.refresh_stats,
            ))
        self.name = params.add(XW.LabelEntry(text='Name', entry_kwargs={'text': '', 'on_text': self.refresh_stats})).entry
        self.mass = params.add(XW.LabelEntry(text='Mass', entry_kwargs={'text': '1', 'on_text': self.refresh_stats})).entry
        self.manuf = params.add(XW.LabelEntry(text='Manuf.', entry_kwargs={'text': '3', 'on_text': self.refresh_stats})).entry
        self.tech = params.add(XW.LabelEntry(text='Tech', entry_kwargs={'text': '3', 'on_text': self.refresh_stats})).entry

        params_frame.add(PTable(
            wname='Component designer periodic table',
            prop_size='180dp', ename_length=1,
            element_callback=self.select_element,
            )).set_size(hintx=0.7)

        if callable(submit_callback):
            params.add(XW.Button(
                text=submit_text,
                on_release=lambda *a: self.submit_callback()))

        self.stats_table = self.add(ComponentStats())

        self.refresh_stats()

    def select_element(self, eindex):
        self.__selected_element = eindex
        self.refresh_stats()

    def refresh_stats(self, *args):
        comp = self.get_component()
        if comp:
            self.stats_table.set_stats(comp.stats)

    def get_component(self):
        if not all([
            floatable(self.mass.text),
            intable(self.manuf.text),
            intable(self.tech.text),
            ]):
            print('Failed to resolve number. Check number parameters.')
            return None
        mass = float(self.mass.text)
        if mass <= 0:
            print('Must have positive mass.')
            return None
        ctype = list(components.COMP_TYPES.keys())[0]
        for ctype_ in components.COMP_TYPES:
            if self.ctype.text == ctype_:
                ctype = ctype_
                break
        name = self.name.text
        material = self.__selected_element
        manufacturing = int(self.manuf.text)
        technology = int(float(self.tech.text))
        comp = self.player.make_component_design(
            ctype, mass, material, manufacturing, technology, name)
        return comp

    def submit_callback(self):
        if callable(self.__submit_callback):
            self.__submit_callback(self.get_component())


class ComponentStats(XW.Box, AppLink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        stats_frame = self.add(XW.Box(orientation='vertical'))
        self.metadata = stats_frame.add(XW.Table(title='Meta'))
        self.base_stats = stats_frame.add(XW.Table(title='Base'))
        self.special_stats = stats_frame.add(XW.Table(title='Special'))

    def set_stats(self, stats):
        if self.app.dev_mode:
            print(make_title('component designer stats'))
            print(adis(stats))
        rounding = 3
        material = stats['metadata']['material']
        ctype = stats['metadata']['ctype']
        ccls = components.COMP_TYPES[ctype]
        self.metadata.apply({
            'Name': stats['metadata']['name'],
            'Type': ctype,
            'Material': material,
            'Manufacturing': stats['metadata']['manufacturing'],
            'Technology': stats['metadata']['technology'],
            })
        self.base_stats.apply({
            'Mass': f"{round(stats['mass'], rounding)} {material}",
            'Maintenance': round(stats['maint_cost'][material], rounding),
            'Build Mat Cost': round(stats['material_cost'][material], rounding),
            'Build Cap Cost': f"{round(stats['build_cost'], rounding)}",
            'Emissions': round(stats['emissions'], rounding),
            })
        specstats = {}
        for i, stat in enumerate(ccls.STATS):
            specstats[components.STAT_NAMES_PRETTY[stat]] = f'{round(stats[stat], rounding)}'
        self.special_stats.apply(specstats)
        self.special_stats.pad(1-i)


class ShipClassView(MainView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        accordion = self.add(XW.Accordion(wname='shipclass view'))
        self.designer_stats = accordion.add_item(
            ShipClassDesigner(
                wname='Ship designer stats',
                submit_callback=self.submit_ship_design),
            title='Design ship class')
        self.browser_stats = accordion.add_item(
            ShipClassBrowser(wname='Ship designer browser stats'),
            title='Existing compare')
        self.custom_designer_stats = accordion.add_item(
            ShipClassDesigner(wname='Ship designer custom compare'),
            title='Custom compare')
        accordion.open_item()

        self.hotkeys.register_dict({
            f'Ship class browser: copy ship class': ('^ c', self.copy_cuid),
            })

    def copy_cuid(self):
        if self.is_visible:
            if isinstance(self.browser_stats._selected_cuid, str):
                kv.Clipboard.copy(self.browser_stats._selected_cuid)

    def submit_ship_design(self, ship_design):
        if ship_design:
            if ship_design.stats['metadata']['name'] == '':
                print('Please enter a name for the ship design.')
                return
            self.player.add_ship_class(ship_design.stats)
        else:
            print('Failed to find a ship design.')


class ShipClassBrowser(XW.Box, AppLink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_cuid = None
        self.orientation = 'horizontal'

        self.design_selector = self.add(XW.TreeView(
            wname='Ship class browser selector',
            callback=self.select_design)).set_size(hinty=0.4)
        self.stats = self.add(ShipClassStats(wname='Ship class browser stats'))
        self.apply_designs()
        self.ucache.register('player_ship_designs', callback=self.apply_designs)

    def apply_designs(self):
        self.design_selector.apply({
            f"<{cuid[:6]}> {self.uni.subclasses[cuid]['stats']['metadata']['name']}": cuid for cuid in self.player.ship_designs
            })

    def select_design(self, cuid):
        self._selected_cuid = cuid
        self.stats.set_stats(self.uni.subclasses[cuid]['stats'])


class ShipClassDesigner(XW.Box, AppLink):
    def __init__(self, submit_callback=None, submit_text=None, **kwargs):
        super().__init__(**kwargs)
        self.__submit_callback = submit_callback
        self.__design_spec = collections.defaultdict(lambda: 0)
        submit_text = 'Submit' if submit_text is None else submit_text
        self.orientation = 'horizontal'

        spec_frame = self.add(XW.Box(orientation='vertical'))
        self.stats_table = self.add(ShipClassStats())

        # Spec frame includes all widgets to edit the current design spec
        comps = spec_frame.add(XW.Box())
        params = spec_frame.add(
            XW.Box(orientation='vertical'))

        self.duids = comps.add(XW.TreeView(callback=self.add_component))
        self.design_spec = comps.add(
            XW.TreeView(callback=self.remove_component))

        # params_subframe = params.add(XW.Box())
        self.name = params.add(XW.LabelEntry(text='Name', lhint=0.3, entry_kwargs={'text': '', 'on_text': self.refresh_stats})).entry
        self.description = params.add(XW.LabelEntry(text='Description', lhint=0.3, entry_kwargs={'text': '', 'on_text': self.refresh_stats})).entry
        params_size = 3
        if callable(submit_callback):
            params.add(XW.Button(
                text=submit_text,
                on_release=lambda *a: self.submit_callback()))
            params_size += 1
        params.set_size(sizey=f'{XW.LINE_DP*params_size}dp')





        self.refresh_stats()

    def remove_component(self, duid):
        self.__design_spec[duid] = minmax(0, float('inf'), self.__design_spec[duid]-1)
        self.refresh_stats()

    def add_component(self, duid):
        self.__design_spec[duid] += 1
        self.refresh_stats()

    def refresh_stats(self, *args):
        self.duids.apply({f"«{duid:0>4}» {_['stats']['metadata']['name']}": duid for duid, _ in self.player.component_designs.items()})
        self.design_spec.apply({f"{count:>2} × «{duid:0>4}» {self.player.component_designs[duid]['stats']['metadata']['name']}": duid for duid, count in self.__design_spec.items() if count > 0})
        ship_design = self.get_ship_design()
        if ship_design:
            self.stats_table.set_stats(ship_design.stats)

    def get_ship_design(self):
        name = self.name.text
        description = self.description.text
        ship_design_spec = self.__design_spec
        if DictOp.sum(ship_design_spec) == 0:
            print(f'Ship design must have at least one component.')
            return None
        ship_design = self.player.make_ship_design(ship_design_spec)
        ship_design.stats['metadata']['name'] = name
        ship_design.stats['metadata']['description'] = name if description == '' else description
        return ship_design

    def submit_callback(self):
        if callable(self.__submit_callback):
            self.__submit_callback(self.get_ship_design())


class ShipClassStats(XW.Box, AppLink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        stats_frame = self.add(XW.Box(orientation='vertical'))
        self.metadata = stats_frame.add(XW.Table(title='Meta'))
        self.maint_costs = stats_frame.add(XW.Table(title='Maintenance'))
        self.material_costs = stats_frame.add(XW.Table(title='Build Cost'))
        self.base_stats = stats_frame.add(XW.Table(title='Base'))
        self.special_stats = stats_frame.add(XW.Table(title='Special'))

    def set_stats(self, stats):
        if self.app.dev_mode:
            print(make_title('component designer stats'))
            print(adis(stats))
        rounding = 3
        self.metadata.apply({
            'Name': stats['metadata']['name'],
            'Description': stats['metadata']['description'],
            })
        self.maint_costs.apply({f'{k}': f'{round(v, rounding)}' for k, v in stats['maint_cost'].items()})
        self.material_costs.apply({
            'Build Cap Cost': f"{round(stats['build_cost'], rounding)}",
            **{f'{k}': f'{round(v, rounding)}' for k, v in stats['material_cost'].items()},
            })
        self.base_stats.apply({
            'Mass': f"{round(stats['mass'], rounding)}",
            'Emissions': round(stats['emissions'], rounding),
            })
        self.special_stats.apply({f'{sname}': f'{round(stats[stat_], rounding)}' for stat_, sname in components.STAT_NAMES_PRETTY.items()})


class PTable(XW.Box, AppLink):
    def __init__(self, ename_length=3, prop_size=None, default_prop=3,
                 prop_callback=None, element_callback=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.__selected_prop = default_prop
        self.__prop_callback = prop_callback
        self.__element_callback = element_callback
        self.__ename_length = ename_length
        if prop_size is None:
            prop_size = '250dp'

        left_frame = self.add(XW.Box(orientation='vertical')
                              ).set_size(sizex=prop_size)
        self.label = left_frame.add(W.Label()).set_size(sizey=XW.LINE_DP_STR)
        self.filters = left_frame.add(
            XW.LabelEntry(text='Filter', entry_kwargs={
                'on_text': lambda *a: self.set_prop()
                })).set_size(sizey=XW.LINE_DP_STR).entry
        self.prop_selector = left_frame.add(XW.TreeView(callback=self.set_prop))

        right_frame = self.add(XW.Box(orientation='vertical'))
        self.plabel = right_frame.add(XW.Label())
        self.ptable_frame = right_frame.add(
            W.GridLayout(cols=self.__selected_prop))
        self.ptable_frame.make_bg((0, 0, 0, 1))
        self.ptable = {e: self.ptable_frame.add(XW.Label(
            no_size=True, on_release=lambda *a, _=ei: self.element_callback(_),
            )) for ei, e in enumerate(common.E.NAMES)}
        seld = {}
        hotk = {}
        for pi, pn in [*enumerate(common.E.PROPERTIES)][2:]:
            pi += 1
            seld[f'{pi}. {pn}'] = pi
            hotk[f'Chemistry: select property {pi}'] = (
                f'! {pi}',
                lambda _=pi: self.set_prop(_))
        self.prop_selector.apply(seld)
        self.hotkeys.register_dict({
            **hotk,
            'Chemistry: select next property': ('! down', self.next_prop),
            'Chemistry: select previous property': ('! up', lambda: self.next_prop(prev=True)),
            })
        self.set_prop()

    def element_callback(self, eindex):
        if callable(self.__element_callback):
            self.__element_callback(eindex)
        r, c = common.E.row_col(eindex, self.__selected_prop)
        q, b = common.ptable(eindex, self.__selected_prop)
        rcs =  f'{r+1}/{c+1}'
        qb = f' - B/Q: {round(q*100, 1)} / {round(b*100, 1)} ({rcs})'
        self.label.text = f'{common.E.NAMES[eindex]}{qb if self.dev_mode else ""}'
        self.set_prop()

    def next_prop(self, prev=False):
        if not self.is_visible:
            return
        new_prop = self.__selected_prop+(-1 if prev else 1)
        self.set_prop(minmax(3, common.E.PCOUNT, new_prop))

    def set_prop(self, prop=None):
        if prop is None:
            prop = self.__selected_prop
        self.__selected_prop = minmax(3, common.E.PCOUNT, prop)
        self.ptable_frame.cols = self.__selected_prop
        self.plabel.text = f'{common.E.PROPERTIES[prop-1]} - {common.E.row_col(common.E.ECOUNT, prop)[0]+1} Quarks × {prop} Bosons'
        filters = common.E.S if self.filters.text == '' else self.filters.text
        for eindex, (ename, btn) in enumerate(self.ptable.items()):
            r, c = common.E.row_col(eindex, prop)
            q, b = common.ptable(eindex, self.__selected_prop)
            if ename[0].lower() in filters:
                btn.text = f'{eindex+1:>2}.{ename[:self.__ename_length]}'
                btn.make_bg(color=(q, eindex/common.E.ECOUNT, b, 0.5), hsv=False)
            else:
                btn.text = ''
                btn.foreground_color = (0, 0, 0)
                btn.make_bg((0, 0, 0, 0))
        if callable(self.__prop_callback):
            self.__prop_callback(self.__selected_prop)


class PlayerView(MainView):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.sim_steps = {
            '1 second': 0.001,
            '1 day': 1,
            '10 days': 10,
            '100 days': 100,
            '1 kiloday': 10**3,
            '10 kilodays': 10**4,
            '100 kilodays': 10**5,
            '1 annum': 10**6,
            }
        sim_step_frame = self.add(XW.Box()).set_size(sizey=XW.LINE_DP_STR)
        sim_step_frame.add(XW.Label(text='Simulate:'))
        for label, step in self.sim_steps.items():
            sim_step_frame.add(XW.Button(
                text=f'{label}',
                on_release=lambda *a, s=step: self.app.simulate_time(s),
                ))

        bottom_frame = self.add(XW.Box())

        details_frame = bottom_frame.add(XW.Box(orientation='vertical')
            ).set_size(sizex='350dp')
        self.uni_details = details_frame.add(XW.Table())
        self.general_table = details_frame.add(XW.Table(title='Player stats'))
        self.maintenance_costs = details_frame.add(XW.Table(title='Maintenance /kd'))
        self.general_label = details_frame.add(W.Label())
        right_frame = bottom_frame.add(XW.Box(orientation='vertical'))
        self.notifications = right_frame.add(XW.TreeView())

        self.ucache.register_calls({
            'refresh_all': self.refresh,
            'time': self.refresh,
            'notifications': self.notify,
            })
        self.hotkeys.register_dict({
            f'Simulation: tick x{s}': (
                f'^+ {i if i>0 else "`"}',
                lambda _=s, *a: self.app.simulate_time(_)
                ) for i, s in enumerate(self.sim_steps.values())
            })

    def notify(self):
        self.app.menu_frame.make_bg(color=(0, 1, 1, .5))
        self.refresh()

    def refresh(self, force=False):
        if not self.is_visible and not force:
            return
        if self.app.dev_mode:
            print(f'Refreshing {self}')

        self.app.menu_frame.make_bg(color=(0, 0, 0, 0))

        self.uni_details.apply({
            'Seed': self.uni.seed.seed,
            'Time': adis(self.uni.time, precision=5),
            'Objects': len(self.uni.bodies),
            '-'*10: '-'*10,
            'Simrate': round(self.uni.simrate.rate, 1),
            })

        self.general_table.apply({
                f'Name': f'{self.player.name}',
                f'Idle ships': f'{len(self.player.idle_ships)} / {len(self.player.ships)}',
            })

        self.maintenance_costs.apply({
            k: adis(v/1000) for k, v in self.player.total_maintenance_costs.items()
            })

        self.general_label.text = sjoin([
            f'Name: {self.player.name}',
            f'Ships: {len(self.player.ships)}',
            make_title('Player dict', 20),
            adis([*self.player.__dict__.keys()], split_threshold=0),
            ])

        self.notifications.apply([adis(_, 120) for _ in self.uni.player.notifications[:10]])


class MapView(MainView):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.__selected_buid = self.uni.player.starting_station

        self.object_filter = BodySelector(wname='Map Object Selector', callback=self.select_body)
        self.map_canvas = MapCanvas(wname='Map Canvas', get_buids=self.object_filter.get_buids)
        self.object_filter.set_refresh_callback(self.map_canvas.refresh)

        # TOP BAR
        top_bar = self.add(XW.Box()).set_size(sizey=XW.LINE_DP_STR)
        top_bar.add(XW.Button(
            text='Galactic center',
            on_release=lambda *a: self.map_canvas.set_view(center=(0, 0))))
        top_bar.add(XW.Button(
            text='Center home',
            on_release=self.center_home))
        top_bar.add(XW.Button(text='Zoom -', on_release=lambda *a: self.map_canvas.zoom_out()))
        top_bar.add(XW.Button(text='Zoom +', on_release=lambda *a: self.map_canvas.zoom_in()))
        self.sensitivity = top_bar.add(XW.LabelEntry(
            text='Emission sens:',
            entry_kwargs={
                'on_text': self.set_sensitivity,
                },
            ))
        self.sensitivity.entry.set_size(sizex=f'{XW.LINE_DP*1.5}dp')
        self.toggles = {
            'labels': [None, 'Labels'],
            'scales': [None, 'Scales'],
            'draw_all': [None, 'Draw all'],
            }
        for param, (cb__, label) in self.toggles.items():
            cb = top_bar.add(XW.CheckButton(
                text=label,
                callback=lambda w, v, p=param: self.map_canvas.set_draw_params(**{p: v})))
            cb.active = active=self.map_canvas.draw_params[param]
            self.toggles[param][0] = cb

        # OBJECT FILTER
        bottom = self.add(XW.Box())
        bottom.add(self.object_filter)

        # MAP CANVAS
        bottom.add(self.map_canvas).set_size(hint=(1, 1))

        self.ucache.register_source('position_anchor', lambda: tuple(self.view_center))
        self.ucache.register_call('refresh_all', self.refresh)
        self.ucache.register_call('time', self.refresh)
        self.ucache.register_call('bodies', self.refresh)
        self.hotkeys.register_dict({
            f'Map: pan up': ('^ up', lambda: self.map_canvas.pan('n')),
            f'Map: pan down': ('^ down', lambda: self.map_canvas.pan('s')),
            f'Map: pan left': ('^ left', lambda: self.map_canvas.pan('w')),
            f'Map: pan right': ('^ right', lambda: self.map_canvas.pan('e')),
            f'Map: center home': ('^ home', self.center_home),
            f'Map: zoom in': ('^ pagedown', self.map_canvas.zoom_in),
            f'Map: zoom out': ('^ pageup', self.map_canvas.zoom_out),
            f'Map: toggle labels': ('^ l', self.toggles['labels'][0].toggle),
            f'Map: toggle scales': ('^ s', self.toggles['scales'][0].toggle),
            f'Map: set emission sensitivity': ('^ e', self.sensitivity.entry.set_focus),
            })

    def center_home(self, *args):
        self.map_canvas.set_view(center=self.uni.moria)

    def set_sensitivity(self, *args):
        try:
            sens = float(self.sensitivity.entry.text)
        except ValueError:
            sens = None
        self.map_canvas.set_draw_params(sensitivity=sens)

    def select_body(self, buid):
        self.map_canvas.set_view(center=self.uni.id2body(buid).position)

    def refresh(self, force=False):
        if not self.is_visible and not force:
            return
        if self.app.dev_mode:
            print(f'Refreshing {self}')
        self.object_filter.refresh(force=force)
        self.map_canvas.refresh(force=force)

    @property
    def view_center(self):
        return self.map_canvas.view_center


class MapCanvas(XW.RCanvas, AppLink):
    def __init__(self, get_buids=None, **kwargs):
        super().__init__(**kwargs)
        self.fps_limit = 5
        self._colors = {
            'bg': (0.6, 1, 0.05),
            'legend': (0, 1, 1),
            }
        self.draw_params = {
            'draw_all': False,
            'labels': True,
            'sensitivity': 0,
            'scales': False,
            }
        self.__last_frame = ping()
        # Zoom Scale equates to real universe distance units per pixel on the canvas
        self.__zoom_scale = 10**2
        self.__real_center = self.uni.moria
        self._update_real_anchor()
        self.__get_buids = self.app.visible_buids if get_buids is None else get_buids
        self.ucache.register_call('refresh_all', self.refresh)
        self.ucache.register_call('time', self.refresh)
        # self.ucache.register_call('bodies', self.refresh_delayed)
        self.ucache.register_call('window_size', self.refresh_delayed)
        self.ucache.register(f'{self}-visible', lambda: self.is_visible, self.refresh_delayed)
        self.refresh(force=True)

    def set_draw_params(self, **d):
        self.draw_params.update(d)
        self.refresh()

    def pan(self, direction, pages=0.1):
        if direction not in 'nsew' or len(direction) != 1:
            raise ValueError(f'Direction must be one of : n s e w')
        x, y = 0, 0
        delta = self.__zoom_scale * min(self.width, self.height) * pages
        if direction == 'n':
            y += delta
        elif direction == 's':
            y -= delta
        elif direction == 'e':
            x += delta
        elif direction == 'w':
            x -= delta
        new_center = self.__real_center[0] + x, self.__real_center[1] + y
        self.set_view(center=new_center)

    def zoom_out(self):
        self.set_view(zoom=self.__zoom_scale * 4/3)

    def zoom_in(self):
        self.set_view(zoom=self.__zoom_scale * 3/4)

    def set_view(self, center=None, zoom=None):
        if zoom is not None:
            self.__zoom_scale = zoom
        if center is not None:
            self.__real_center = center
        # Caching the real anchor (real universe coordinates of our (0, 0) point
        #    on the canvas) will help us with performance later when translating
        #    real universe coords to canvas coords.
        self._update_real_anchor()
        self.refresh(force=True)

    def _update_real_anchor(self):
        rx = self.width * self.__zoom_scale / 2
        ry = self.height * self.__zoom_scale / 2
        self.__real_anchor = self.__real_center[0] - rx, self.__real_center[1] - ry

    def refresh(self, force=False):
        if not force:
            if not self.is_visible:
                return
        self.__last_frame = ping()
        self._update_real_anchor()
        self.draw_map()

    def refresh_delayed(self):
        kex.Clock.schedule_once(lambda dv: self.refresh(), 0.2)

    def draw_map(self, debug=False):
        if debug:
            print(f'Redrawing: {self}')
        buids = self.__get_buids()
        # FULL WIPE
        if debug:
            print(f'Wiping...')
        self.canvas.clear()
        # FULL DRAW
        if debug:
            print(f'Drawing...')
        with self.canvas:
            # BACKGROUND
            self.draw_fill(self._colors['bg'])
            kv.HColor(*self._colors['legend'])
            # DEBUG CROSS
            if self.dev_mode:
                kv.Line(points=(0, 0, self.width, self.height), dash_length=3, dash_offset=15)
                kv.Line(points=(0, self.height, self.width, 0), dash_length=3, dash_offset=15)
            # CENTER CROSSHAIR
            # self.draw_crosshair(*self.canvas_center)
            # BODIES
            # Find bounds of canvas in real coordinates
            bound_l, bound_b = self.pix2real(0, 0)
            bound_r, bound_t = self.pix2real(self.width, self.height)
            body_name_labels = list()
            for buid in buids:
                b = self.uni.id2body(buid)
                # Filter if the body is not in the canvas pixels
                if any([
                    b.position[0] < bound_l,
                    b.position[0] > bound_r,
                    b.position[1] < bound_b,
                    b.position[1] > bound_t,
                    ]) and not self.draw_params['draw_all']:
                    continue
                label = self.draw_body(b)
                if label:
                    body_name_labels.append(label)
            self.draw_labels(body_name_labels)
            # MAP SCALE
            XW.CanvasLabel(text=adis(self.__zoom_scale*100),
                           pos=(10, 10))
            kv.Line(points=(5, 5, 105, 5))
            if self.draw_params['scales']:
                # Draw concentric circles around center, with a scale for each radius
                scount = 6
                radius_unit = max(self.width, self.height) / (scount*2)
                axis = 0 if self.width>self.height else 1
                for ri in range(scount):
                    radius = radius_unit*(ri+1)
                    kv.Line(circle=(*self.canvas_center, radius), dash_length=15, dash_offset=10)
                    x, y = self.canvas_center
                    y -= 15
                    t = adis(self.__zoom_scale*radius)
                    XW.CanvasLabel(text=t, pos=(x-radius, y))
                    XW.CanvasLabel(text=t, pos=(x+radius, y))
                    XW.CanvasLabel(text=t, pos=(x, y-radius))
                    XW.CanvasLabel(text=t, pos=(x, y+radius+25))
        if self.app.dev_mode:
            print(f'.canvas redraw complete: {self}.')

    def draw_body(self, b):
        # Point/circle
        color = self.uni.subclass2color(b.subclass)
        kv.HColor(*color)
        x, y = self.real2pix(*b.position)
        # size = minmax(1, 6, math.log(b.dry_mass)/2)
        size = 4
        kv.Ellipse(pos=(x-(size/2), y-(size/2)), size=(size, size))
        # Emissions
        if self.draw_params['sensitivity']:
            sens = self.draw_params['sensitivity']
            radius = common.Sensing.sensor_range(b.dry_mass, sens)
            kv.Line(circle=(x, y, radius/self.__zoom_scale))
        # Name label
        if self.draw_params['labels']:
            # XW.CanvasLabel(text=b.name, pos=(x+size+5, y+size), color=color)
            l = ((x+size+5, y+size), color, b.name)
            return l
        return None

    def draw_labels(self, labels):
        max_line_count = 4
        draw_labels = {}
        for pos, color, text in labels:
            if str(pos) in draw_labels:
                # draw_labels[str(pos)][1] = (0, 0, 1)
                draw_labels[str(pos)][2].append(text)
            else:
                draw_labels[str(pos)] = [pos, color, [text]]

        for pos, color, texts in draw_labels.values():
            if len(texts) > max_line_count:
                text = '\n'.join(texts[:max_line_count-1])+f'\n... +{len(texts)-max_line_count+1}'
            else:
                text = '\n'.join(texts)
            XW.CanvasLabel(text=text, pos=pos, color=color)

    def real2pix(self, real_x, real_y):
        real_offset = real_x - self.__real_anchor[0], real_y - self.__real_anchor[1]
        pix_offset = real_offset[0] / self.__zoom_scale, real_offset[1] / self.__zoom_scale
        return pix_offset

    def pix2real(self, pix_x, pix_y):
        return self.__real_anchor[0] + pix_x*self.__zoom_scale, self.__real_anchor[1] + pix_y*self.__zoom_scale

    @property
    def view_center(self):
        return self.__real_center

    @property
    def view_zoom(self):
        return self.__zoom_scale


class BodyView(MainView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__selected_buid = None
        self.selector = self.add(BodySelector(
            wname='Body Selector', callback=self.set_buid,
            list_kwargs={'auto_invoke': True},
            ))
        self.viewer = self.add(XW.ScreenSwitch(
            wname='Body Viewer ScreenSwitch',
            screens=['rock', 'ship'],
            transition=kv.FadeTransition(duration=0.1)
            )).set_size(hintx=0.7)
        self.rock_viewer = self.viewer.get_screen('rock').add(
            BodyRockViewer(wname='Rock Viewer'))
        self.ship_viewer = self.viewer.get_screen('ship').add(
            BodyShipViewer(wname='Ship Viewer'))
        self.rename_pop = XW.PopupEntry(
            title='Rename object',
            callback=self.rename_body,
            prefill=self.rename_prefill)

        # Select the first buid in our selector - ensure we have a selected buid
        self.selector.select_first()
        self.ucache.register_call('time', self.refresh)
        self.ucache.register_call('refresh_all', self.refresh)
        self.hotkeys.register_dict({
            f'Object broswer: rename object': ('^ r', self.pop_rename_body),
            f'Object broswer: copy buid': ('^ c', self.copy_buid),
        })

        self.refresh(force=True)

    def set_buid(self, buid):
        self.__selected_buid = buid
        self.refresh()

    def refresh(self, force=False):
        if (not self.is_visible and not force) or not self.__selected_buid:
            return
        if self.app.dev_mode:
            print(f'Refreshing {self}')
        self.selector.refresh(force=force)
        buid = self.__selected_buid
        b = self.uni.id2body(buid)
        if b.is_rock:
            self.viewer.switch_screen('rock')
            self.rock_viewer.set_buid(buid)
        if b.is_ship:
            self.viewer.switch_screen('ship')
            self.ship_viewer.set_buid(buid)

    def copy_buid(self):
        if self.is_visible:
            kv.Clipboard.copy(self.__selected_buid)

    def rename_prefill(self, *args):
        return self.uni.id2body(self.__selected_buid).name

    def rename_body(self, name):
        self.uni.id2body(self.__selected_buid).name = name
        self.refresh()
        self.selector.refresh(force=True)

    def pop_rename_body(self, *args):
        if self.is_visible:
            self.rename_pop.open()


class BodyShipViewer(W.GridLayout, AppLink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 3
        self.details = self.add(
            XW.Table(title='Details', ksize=0.5)
            )  #.set_size(hintx=0.33, hinty=0.5)

        engineering = self.add(
            XW.Box(orientation='vertical')
            )  #.set_size(hintx=0.33, hinty=0.5)
        self.engineering_stats = engineering.add(
            XW.Table(title='Engineering'))
        self.repair_costs = engineering.add(
            XW.Table(title='Repair costs'))
        self.material_costs = engineering.add(
            XW.Table(title='Build costs'))

        self.drive_stats = self.add(
            XW.Table(title='Drive', ksize=0.6)
            )  #.set_size(hintx=0.33, hinty=0.5)

        cargo = self.add(
            XW.Box(orientation='vertical')
            )  #.set_size(hintx=0.33, hinty=0.5)
        self.cargo_stats = cargo.add(XW.Table(title='Cargo', ksize=0.6))
        self.cargo_hold = cargo.add(XW.Table(title='Cargo Hold', ksize=1.2))

        self.production_stats = self.add(
            XW.Table(title='Production')
            )  #.set_size(hintx=0.33, hinty=0.5)

        bridge = self.add(
            XW.Box(orientation='vertical')
            )  #.set_size(hintx=0.33, hinty=0.5)
        self.bridge_stats = bridge.add(XW.Table(title='Bridge', ksize=0.6))

    def set_buid(self, buid):
        b = self.uni.id2body(buid)
        assert b.is_ship

        # Details
        details = {
            'Name': b.name,
            'Class': f'{b.resolve_subclass["name"]}',
            'Tags': f'{b.bridge.tags_formatted()}',
            'Description': f'{b.resolve_subclass["description"]}',
            'Mass': adis(b.mass),
            'Position': format_position(b.position),
            'Distance': adis(universe.vmag(b.position-self.ucache.position_anchor.ref)),
            'Velocity': adis(b.velocity),
            'sensor_stack': len(b._sensor_callback_stack),
            'buid': f'{b.buid}',
            'cuid': f'{b.subclass}',
            }
        self.details.apply(details)

        # Bridge
        self.bridge_stats.apply({
            'Tags': f'{b.bridge.tags_formatted()}',
            'Status': f'{b.bridge.status}',
            'Current order': f'{b.bridge.current_order.summary}',
            'Current state': f'{b.bridge.current_order.current_state}',
            **{adis(k): diss(v, 30) for k, v in b.bridge.current_order.dcache.items()}
            })

        # Engineering
        self.engineering_stats.apply({
            'Service time': f'{format_time(b.service_time)}',
            f'Condition': f'%{round(b.maintenance_condition*100, 3)}',
            f'Remaining service': f'{format_time(b.remaining_service_time)}',
            'Degrade rate': f'%{round(100000/b.service_time, 3)} / kd',
            f'Engineering': f'{adis(b.engineering)} /{adis(b.dry_mass)}',
            })
        repair_cost = b.repair_cost(1)
        self.repair_costs.apply({k: f'{adis(repair_cost[k])} /{adis(v)}' for k, v in b.stats['maint_cost'].items()})
        self.material_costs.apply({
            'Capacity': f"{round(b.stats['build_cost'], 2)}",
            **{k: adis(v) for k,v in b.stats['material_cost'].items()}
            })

        # Production
        self.production_stats.apply({
            'Mining': f'{round(b.stats["mining_capacity"], 3)}',
            'Build': f'{round(b.stats["build_capacity"], 3)}',
            'Sensitivity': f'{round(b.stats["sensitivity"], 3)}',
            'Fuel burned': f'{adis(b.service_stats["fuel_consumed"], 3)}',
            '-'*10: '-'*10,
            **{f'{stat_name}': f'{adis(b.stats[stat_])}' for stat_, stat_name in components.STAT_NAMES_PRETTY.items()},
            })

        # Drive
        tug_str = f'{adis(b.tugging_mass)}'
        if len(b._tugging) == 1:
            tug_str = f'{adis(b.tugging_mass)} ({self.app.uni.id2body(list(b._tugging)[0]).name})'
        elif len(b._tugging) > 1:
            tug_str = f'{adis(b.tugging_mass)} ({len(b._tugging)} ships)'
        self.drive_stats.apply({
            'dV available': f'{adis(b.available_dv)} / {adis(b.total_dv)}',
            'Fuel tank': f'{adis(b.fuel_tank)} / {adis(b.stats["fuel_tank"])}',
            'dV cost': f'{adis(b.deltav_fuel_cost(1))} /{adis(b.deltav_fuel_cost(1, use_mass=b.full_wet_mass))}',
            'Drive warmup': f'{adis(b.drive_warmup_time(1))}',
            'Drive mass': f'{adis(b.drive_mass)}',
            'Isp': f'{adis(b.stats["isp"])}',
            'Thrust': f'{adis(b.stats["thrust"])}',
            'Docked': f'{self.app.uni.id2body(b._docked).name}' if b._docked is not None else 'No',
            'Tugging': f'{tug_str}',
            })

        # Cargo
        cargo_hold = {}
        for item, amount in sorted(b.cargo_hold.items(), key=lambda x: -x[1]):
            if amount <= 0:
                continue
            cargo_hold[item] = [round(amount, 3)]
        self.cargo_hold.apply(cargo_hold)
        self.cargo_stats.apply({
            'Dry mass': f'{adis(b.dry_mass)}',
            'Cargo mass': f'{adis(b.cargo_mass)} / {adis(b.stats["hold"])}',
            'Fuel tank': f'{adis(b.fuel_tank)} / {adis(b.stats["fuel_tank"])}',
            'Wet mass': [f'{adis(b.wet_mass)}', f'/{adis(b.full_wet_mass)}'],
            })


class BodyRockViewer(XW.Basket, AppLink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.orientation = 'vertical'
        self.details = self.add(XW.Table(title='Details', ksize=0.5)).set_size(hintx=.33)
        self.elements = self.add(XW.Table(
            title='Elements', ksize=1.5, columns=5,
            legend=['Element', 'Cap', 'Acc', 'Cost', 'Sat', 'Quant'],
            )).set_size(hintx=.66)

    def set_buid(self, buid):
        b = self.uni.id2body(buid)
        assert b.is_rock

        # DETAILS
        system = '▒ Galaxy'
        if b.parent:
            system = f'{b.parent.name}'
            if b.parent.parent:
                system += f' - {b.parent.parent.name}'
            else:
                system += f' - ▒ Galaxy'
        self.details.apply({
            'Name': b.name,
            'Class': f'{b.resolve_subclass["name"]}',
            'System': system,
            'Description': f'{b.resolve_subclass["description"]}',
            'Mass': adis(b.mass),
            'Position': format_position(b.position),
            'Distance': adis(universe.vmag(b.position-self.ucache.position_anchor.ref)),
            'Velocity': adis(b.velocity),
            'buid': f'{b.buid}',
            'cuid': f'{b.subclass}',
            })

        # ELEMENTS
        # TODO allow for sorting options
        if b.buid in self.uni.player.rock_database and self.uni.player.rock_database[b.buid]['survey']:
            elements = tuple(sorted(list(b.elements.keys()), key=lambda x: -common.E.NAMES.index(x)))
            elements_table = {}
            rquant = {}
            for ename in elements:
                if not b.has_element(ename):
                    continue
                cap = b.elements[ename]['capacity']
                acc = b.elements[ename]['accessibility']
                elements_table[ename] = [
                    round(cap*1000, 3),  # capacity
                    f'% {round(acc*100, 1)}',  # access
                    f'x {round(1/acc, 2)}',  # cap cost
                    round(cap*1000/acc, 3),  # saturation
                    f'{round(b.get_realq(ename), 1)}',  # rquant - dev
                    ]
        else:
            elements_table = {'Unsurveyed.': []}
            self.uni.debug(sjoin([
                make_title(f'Hidden elements of {b.fname}'),
                adis(b.elements),
                ]))

        self.elements.apply(elements_table)


class OrdersView(MainView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__selected_bridge = None
        self.__param_widgets = {}

        self.bridge_selector = self.add(BodySelector(
            wname='Bridge Selector',
            get_buids=lambda: self.uni.player.ships,
            default_sorting='Class',
            callback=self.set_bridge,
            list_kwargs={'auto_invoke': True},
            ))
        bridge_frame = self.add(
            XW.Box(orientation='vertical')).set_size(hintx=0.2)
        param_frame = self.add(XW.Box()).set_size(hintx=0.3)
        current_order_frame = self.add(XW.Box(orientation='vertical')).set_size(hintx=0.3)

        # Order selector and Bridge summary
        self.summary_table = bridge_frame.add(
            XW.Table(ksize=0.5)).set_size(hinty=0.5)

        self.set_tags_pop = XW.PopupEntry(
            title='Set tags',
            callback=self.apply_tags,
            prefill=self.tags_prefill)
        bridge_frame.add(XW.Button(
            text='Set tags', on_release=self.set_tags_pop_open))

        self.order_selector = bridge_frame.add(
            XW.TreeView(callback=self.make_order, wname=f'order selector')
            ).apply({_: _ for _ in agency.ORDER_TYPES.keys()})

        # Parameter frame
        self.param_switch = param_frame.add(
            XW.ScreenSwitch(wname=f'order param screenmanager'))

        # Order param screens - make a screen for each order type
        self.submit_buttons = {}
        for order_name, order_class in agency.ORDER_TYPES.items():
            screen = self.param_switch.add_screen(order_name)
            frame = screen.add(XW.Box(orientation='vertical'))
            frame.add(XW.Button(
                text=f'{order_name} (click to reset)',
                on_release=lambda *a, o=order_name: self.reset_params(o),
                ))
            param_subframe = frame.add(W.GridLayout(cols=2))
            self.__param_widgets[order_name] = {}
            # For each parameter in the order specification, add the widget and store in our param widgets dictionary
            for param, (ptype, default, label) in order_class.PARAMS.items():
                widget = ParamGUI.widgets[ptype](label=label)
                self.__param_widgets[order_name][param] = widget
                widget.set(default)
                param_subframe.add(widget)
                param_subframe.add(XW.Button(
                    text='»', on_release=lambda *a, _=param: self.order_update(_)
                    )).set_size(sizex='15dp', sizey=XW.LINE_DP_STR)
            # Debug param
            widget = ParamGUI.widgets['bool'](label='Debug')
            self.__param_widgets[order_name]['debug'] = widget
            widget.set(False)
            param_subframe.add(widget)
            param_subframe.add(XW.Button(
                text='»', on_release=lambda *a: self.order_update('debug')
                )).set_size(sizex='15dp', sizey=XW.LINE_DP_STR)
            # Submit button
            b1 = frame.add(XW.Button(
                text=f'Submit {order_name} order',
                on_release=lambda *a, o=order_name: self.give_order(o),
                ))
            b2 = frame.add(XW.Button(
                text=f'Queue {order_name} order',
                on_release=lambda *a, o=order_name: self.give_order(o, queue=True),
                ))
            self.submit_buttons[order_name] = (b1, b2)

        # Current order frame - showing status of current order
        self.current_order_table = current_order_frame.add(
            XW.Table(title='Current Order'))

        self.bridge_selector.select_first()
        self.hotkeys.register(
            'Orders view: set tags', '^+ t', self.set_tags_pop_open)
        self.ucache.register_call('time', self.refresh)

    def tags_prefill(self, *args):
        b = self.uni.id2body(self.__selected_bridge)
        return ';'.join(b.bridge.tags)

    def set_tags_pop_open(self, *args):
        if self.is_visible:
            self.set_tags_pop.open()

    def apply_tags(self, text, *args):
        tags = set(text.split(';'))
        b = self.uni.id2body(self.__selected_bridge).bridge
        b.set_tags()
        b.set_tags(tags)
        self.set_tags_pop.dismiss()
        self.refresh()

    def order_update(self, param):
        bridge = self.uni.id2body(self.__selected_bridge).bridge
        oname = self.param_switch.current_screen.name
        v = self.__param_widgets[oname][param].get()
        if v is None:
            v = agency.ORDER_TYPES[oname].PARAMS[p][1]
        bridge.update_order({param: v})
        self.refresh()

    def reset_params(self, order_name):
        order_class = agency.ORDER_TYPES[order_name]
        for param, (ptype, default, label) in order_class.PARAMS.items():
            self.__param_widgets[order_name][param].set(default)

    def make_order(self, order_name):
        self.param_switch.switch_screen(order_name)

    def give_order(self, order_name, queue=False):
        order_class = agency.ORDER_TYPES[order_name]
        params = {'debug': True}
        if self.app.dev_mode:
            print(f'{"Queueing" if queue else "Giving"} order:')
        for p, w in self.__param_widgets[order_name].items():
            v = w.get()
            params[p] = order_class.PARAMS[p][1] if v is None else v
            if self.app.dev_mode:
                print(f'{p}: {v}')
        bridge = self.uni.id2body(self.__selected_bridge).bridge
        bridge.give_orders(orders=[
            (order_name, params),
            ], queue=queue)
        self.refresh()

    def set_bridge(self, buid):
        self.__selected_bridge = buid
        b = self.uni.id2body(buid)
        self.summary_table.apply({
            'Bridge': f'{b.name}',
            'Tags': f'{b.bridge.tags_formatted()}',
            'Status': f'{b.bridge.status}',
            })
        self.current_order_table.apply({
            'Status': f'{b.bridge.status}',
            'Current order': f'{b.bridge.current_order.summary}',
            'Current state': f'{b.bridge.current_order.current_state}',
            f"{'-'*5} ": '-'*5,
            **{adis(k): diss(v, 30) for k, v in b.bridge.current_order.dcache.items()},
            f"{'-'*5}": '-'*5,
            **{adis(k): diss(v, 30) for k, v in b.bridge.current_order.params.items()},
            })
        for oname, (b1, b2) in self.submit_buttons.items():
            b1.text = f'Submit {oname} to {adis(b.name, max_length=20)}'
            b2.text = f'Queue {oname} to {adis(b.name, max_length=20)}'

    def refresh(self, force=False):
        if not force and not self.is_visible:
            return
        self.bridge_selector.refresh(force=force)
        self.set_bridge(self.__selected_bridge)


class BodySelector(XW.Box, AppLink):
    def __init__(self,
                 get_buids=None,
                 callback=None,
                 default_sorting=None,
                 refresh_callback=None,
                 default_size=True,
                 list_kwargs=None,
                 **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        SIZEX = '100dp'
        SIZEX_WIDE = '400dp'
        self.set_size(sizex=SIZEX_WIDE)
        self.__cached_buids = []
        self.__get_buids = self.app.get_visible_buids if get_buids is None else get_buids
        self.__refresh_callback = refresh_callback
        self.sorting_keys = {
            'Distance': lambda x: self.app.uni.sort_by_distance_key(buid=x, target=self.uni.moria),
            'Class': self.uni.sort_by_subclass_key,
            'Mass': self.uni.sort_by_mass_key,
            }
        self.show_rocks = True
        self.show_ships = True
        # Class Filter
        class_filter_frame = self.add(XW.Box()).set_size(sizey=XW.LINE_DP_STR)
        class_filter_frame.add(W.Label(text='Class:')).set_size(sizex='60dp')
        self.class_filter = class_filter_frame.add(XW.Entry(
            wname='Class filter',
            on_text=lambda *a: self.app.ucache[f'{self}-cf'].tvar.flag(),
            on_text_validate=lambda *a: self.select_current(),
            defocus_on_validate=False,
            ).set_size(sizey=XW.LINE_DP_STR))
        # Name Filter
        name_filter_frame = self.add(XW.Box()).set_size(sizey=XW.LINE_DP_STR)
        name_filter_frame.add(W.Label(text='Name:')).set_size(sizex='60dp')
        self.name_filter = name_filter_frame.add(XW.Entry(
            wname='Name filter',
            on_text=lambda *a: self.app.ucache[f'{self}-nf'].tvar.flag(),
            on_text_validate=lambda *a: self.select_current(),
            defocus_on_validate=False,
            ).set_size(sizey=XW.LINE_DP_STR))
        # Count label
        self.count_label = self.add(XW.Label()).set_size(sizey=XW.LINE_DP_STR)
        # Sorting
        self.reverse_sorting = self.add(
            XW.CheckButton(text='Reverse')
            ).set_size(sizey=XW.LINE_DP_STR, sizex=SIZEX)
        self.sorting = self.add(
            XW.RadioSelection(values=self.sorting_keys)
            ).set_size(sizey=f'{XW.LINE_DP*3}dp', sizex=SIZEX)
        if default_sorting:
            self.sorting.set_active_label(default_sorting)
        # Body List
        list_kwargs = {} if list_kwargs is None else list_kwargs
        self.blist = self.add(XW.TreeView(callback=callback, **list_kwargs))

        self.ucache.register_call('refresh_all', self.refresh)
        self.ucache.register_call('bodies', self.refresh)
        self.ucache.register(f'{self}-filters', lambda: str(self.filter_sum()), self.refresh)
        self.hotkeys.register_dict({
            f'Object selector: search by name': ('^ f', self.focus_name),
            f'Object selector: search by class': ('^ g', self.focus_class),
            f'Object selector: select': ('! enter', self.select_current),
            f'Object selector: select next': ('! down', self.select_next),
            f'Object selector: select previous': ('! up', lambda: self.select_next(previous=True)),
            })
        self.refresh(force=True)

    def select_current(self):
        if self.is_visible:
            self.blist.select_current()

    def select_next(self, previous=False):
        if self.is_visible:
            self.blist.select_next(previous=previous)

    def focus_name(self):
        if self.is_visible:
            self.name_filter.set_focus()

    def focus_class(self):
        if self.is_visible:
            self.class_filter.set_focus()

    def set_refresh_callback(self, callback):
        self.__refresh_callback = callback

    def filter_sum(self):
        return [
            self.sorting.value,
            self.reverse_sorting.active,
            self.name_filter.text,
            self.class_filter.text,
            ]

    def select_first(self, *args):
        self.blist.item_select(0)

    def refresh(self, *args, force=False):
        if not self.is_visible and not force:
            return
        if self.app.dev_mode:
            print(f'Refreshing {self}')
        if force:
            self.__cached_buids = tuple()
        new_buids = self.refresh_buids()
        if new_buids != self.__cached_buids:
            self.__cached_buids = new_buids
            self.count_label.text = f'Found {len(self.__cached_buids)} objects'
            name_list = {}
            for buid in self.__cached_buids:
                new_name = f'{self.uni.id2body(buid).name}'
                while new_name in name_list:
                    new_name += ' '
                name_list[new_name] = buid
            self.blist.apply(name_list)
        if self.__refresh_callback:
            self.__refresh_callback()

    def refresh_buids(self):
        cuid_texts = []
        class_filter_text = self.class_filter.text
        if class_filter_text != '':
            for cuid_text in class_filter_text.split(';'):
                if cuid_text != '' and cuid_text != ' ':
                    cuid_texts.append(cuid_text)
        filter_names = []
        names_filter_text = self.name_filter.text
        if names_filter_text != '':
            for name in names_filter_text.split(';'):
                if name != '':
                    filter_names.append(name)

        sorting_key = self.sorting.value

        s = sorted(self.app.uni.filter_bodies(
            self.__get_buids(),
            rocks=self.show_rocks,
            ships=self.show_ships,
            subclasses=cuid_texts if len(cuid_texts) > 0 else None,
            names=filter_names if len(filter_names) > 0 else None,
            ), key=sorting_key)
        if self.reverse_sorting.active:
            s = [*reversed(s)]
        return s

    def get_buids(self):
        return self.__cached_buids


class ParamGUI:
    class ParamBase(XW.Basket):
        def __init__(self, label, **kwargs):
            super().__init__(**kwargs)
            self._label = label
            self.set_size(sizey=XW.LINE_DP_STR)

    class String(ParamBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__value = self.add(XW.LabelEntry(text=self._label)).entry

        def set(self, value):
            self.__value.text = str(value)

        def get(self):
            return self.__value.text

    class Float(String):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def get(self):
            try:
                return float(super().get())
            except ValueError:
                return 0.0

    class Bool(ParamBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cb = self.add(
                XW.CheckButton(text=self._label, label_left=True))

        def set(self, value):
            self.cb.active = value

        def get(self):
            return self.cb.active

    class Element(String):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def get(self):
            return [common.E.NAMES[lists.letter_index(e)] for e in super().get()]

    class StrList(String):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def get(self):
            s = super().get()
            if s == '':
                return []
            return list(super().get().split(';'))

    widgets = {
        'string': String,
        'float': Float,
        'bool': Bool,
        'element': Element,
        'slist': StrList,
        'position': String,
        }


def dis_maint(maint, build=None):
    if build is None:
        return sjoin(f'{adis(round(v, 3))} {ename}' for ename, v in maint.items())
    # This assumes that for every element in build cost there is in maintenance and vice versa
    return sjoin(f'{adis(round(v, 3))} / {adis(build[ename], precision=1)} {ename}' for ename, v in maint.items())


def format_time(t):
    annum = int(t // 1000000)
    kilo = int((t % 1000000) // 1000)
    second = int(t % 1000)
    micro = int((t % 1) * 1000)
    s = [f'{_:>3}' for _ in (annum, kilo, second)]
    return f'{sjoin(s, split="|")}.{micro:0>3}'


def format_position(pos):
    return f'{adis(pos[0], precision=1)},{adis(pos[1], precision=1)}'
