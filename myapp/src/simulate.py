from bokeh.io import curdoc
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Button, Slider, Div, LabelSet
from bokeh.layouts import column, layout, row
import pandas as pd
import numpy as np
import time


class EpidemicSimulation:
    '''
    This code is inspired by this video (https://www.youtube.com/watch?v=gxAaO2rsdIs&t=311s),
    in which a SIR model is implemented.
    We assumed a random walk of every one in a box and redirect their way when hitting the edge of box
    '''
    def __init__(self):
        self.box_size = 30
        self.pop_size = 100  # population size
        self.init_infect = 1  # infected people size
        self.infection_radius = 2   #
        self.infection_rate = 0.9  # < 1
        self.removed_steps = 15
        self.total_steps = 60
        self.sus_color = '#99ccff'              # '#0F76BD'
        self.infect_color = '#ff6666'   #'#BD0F0F'
        self.removed_color = '#9C9A9A'
        self.step_index = 0
        self.step_size = 1.
        self.percent = "0%"
        np.random.seed(100)

        # vars
        self.people = None
        self.data = None
        self.button_start = None
        self.button_stop = None
        self.button_reset = None
        self.layout = None
        self.desc = None
        self.people_source, self.data_source = [], []
        self.tmp_people_x = None
        self.tmp_people_y = None
        self.tmp_people_df = None
        self.tmp = None
        self.start_flag = False
        self.early = False
        self.early_step = None
        self.pre_cal_step = 0

        self.vars = dict()
        self.vars['pop_size'] = self.pop_size
        self.vars['step_size'] = self.step_size
        self.vars['infection_radius'] = self.infection_radius
        self.vars['infection_rate'] = self.infection_rate
        self.vars['total_steps'] = self.total_steps
        self.vars['removed_steps'] = self.removed_steps

        # exe
        self.__call__()

    def __call__(self, *args, **kwargs):
        self._get_buttons()
        self._get_init_data_source()
        self._update_buttons()
        self._get_point_fig()
        self._get_line_fig()
        self._get_legend()
        self._get_plot()

    # def _init_status(self):
    #     self._get_init_data_source()
    #     self._get_point_fig()
    #     self._get_line_fig()
    #     self._get_plot()

    def _update_buttons(self):
        self.slider_pop.on_change('value_throttled', self._update_slider('pop_size', self.slider_pop, self.people))
        self.slider_radius.on_change('value_throttled', self._update_slider('infection_radius', self.slider_radius))
        self.slider_infect_rate.on_change('value_throttled', self._update_slider('infection_rate', self.slider_infect_rate))
        self.slider_total.on_change('value_throttled', self._update_slider('total_steps', self.slider_total))
        self.slider_removed.on_change('value_throttled', self._update_slider('removed_steps', self.slider_removed))
        self.silder_step_size.on_change('value_throttled', self._update_slider('step_size', self.silder_step_size))

    def _get_buttons(self):
        self.button_start = Button(label="Start")
        self.button_start.on_click(self._period_update)
        # self.button_start.on_click(self._update)

        self.button_stop = Button(label="Stop")
        self.button_stop.on_click(self._stop_period)

        self.button_reset = Button(label='Reset')
        self.button_reset.on_click(self._reset)

        self.button_replay = Button(label='Replay')
        self.button_replay.on_click(self._replay)

        self.button_stop_run = Button(label='Cancel', width=100)
        self.button_stop_run.on_click(self._stop_run)

        self.slider_pop = Slider(title='Population Size', value=self.pop_size,
                                 start=50, end=500,
                                 step=1, width=100, width_policy='fit', show_value=True,
                                 )
        self.silder_step_size = Slider(title='Step Size', value=self.step_size,
                                 start=0., end=5.,
                                 step=0.1, width=100, width_policy='fit', show_value=True,
                                 )
        self.slider_radius = Slider(title='Infection Radius', value=self.infection_radius,
                                    start=1, end=10,
                                    step=1, width=100, width_policy='fit', show_value=True,
                                   )
        self.slider_infect_rate = Slider(title='Infection Rate', value=self.infection_rate,
                                         start=0.1, end=1.,
                                         step=0.1, width=100, width_policy='fit', show_value=True,
                                         )
        self.slider_total = Slider(title='Total Simulation Days', value=self.total_steps,
                                   start=30, end=200,
                                   step=1, width=100, width_policy='fit', show_value=True,
                                   )
        self.slider_removed = Slider(title='Removed Days', value=self.removed_steps,
                                     start=30, end=200,
                                     step=1, width=100, width_policy='fit', show_value=True,
                                     )

    def _create_people(self):
        people = dict()
        people['x'] = np.random.uniform(low=1, high=self.box_size - 1, size=self.vars['pop_size'])
        people['y'] = np.random.uniform(low=1, high=self.box_size - 1, size=self.vars['pop_size'])
        people['color'] = [self.sus_color] * (self.vars['pop_size'] - self.init_infect) + [self.infect_color]
        people['infect_days'] = [0] * (self.vars['pop_size'] - self.init_infect) + [1]
        people['ID'] = list(np.arange(self.vars['pop_size']))
        return ColumnDataSource(people)

    def _update_slider(self, target_name, source_, datasource=None):
        def update_col_slider(attrname, old, new):
            self.vars[target_name] = source_.value
            if datasource is not None:
                datasource.data = self._create_people().data
        return update_col_slider

    def _create_data(self):
        data = dict()
        data['time'] = [self.step_index]
        data['Susceptible'] = [self.pop_size - self.init_infect]
        data['Infectious'] = [self.init_infect]
        data['Removed'] = [0]
        return ColumnDataSource(data)

    def _get_init_data_source(self):
        self.people = self._create_people()
        self.data = self._create_data()
        self.text_s = ColumnDataSource(data=dict(
            x=[50],
            y=[50],
            text=['processing {}'.format(self.percent)]))

    def _get_point_fig(self):
        self.point_fig = figure(
                        title='Random People Walk', sizing_mode='scale_width',
                        x_range=[-1, self.box_size+1], y_range=[-1, self.box_size+1],
                        plot_height=200, plot_width=200)
        self.point_fig.circle(source=self.people, x='x', y='y', color='color', size=5)
        self.point_fig.xgrid.grid_line_color = None
        self.point_fig.ygrid.grid_line_color = None

    def _get_line_fig(self):
        self.line_fig = figure(
                        title='SIR Model', sizing_mode='scale_width', #y_range=[-0.1, self.vars['pop_size'] + 5],
                        plot_height=200, plot_width=200,
                        x_range=[0, self.vars['total_steps']])
        self.line_fig.varea_stack(['Infectious', 'Susceptible', 'Removed'], x='time', source=self.data,
                                  color=(self.infect_color, self.sus_color, self.removed_color), alpha=0.6)
        self.line_fig.xgrid.grid_line_color = None
        self.line_fig.ygrid.grid_line_color = None

    def _pre_calculate(self):
        self.tmp = self.data
        self._pre_cal_core()
        return None

    def _pre_cal_core(self):
        if self.pre_cal_step > self.vars['total_steps']:
            if self.early:
                self.run_step_limit = self.early_step - 1
            else:
                self.run_step_limit = self.vars['total_steps'] - 1
            self._display()
            self.callback_id = curdoc().add_periodic_callback(self._run, 50)
        else:
            people = self._update_people_pos(self.tmp_people_x, self.tmp_people_y)
            num_sus, num_infec, num_rem, self.tmp_people_df = self._cal_data(people, self.tmp_people_df)
            people = ColumnDataSource(self.tmp_people_df.to_dict(orient='list')).data
            self.people_source.append(people)
            self.tmp_people_x = people['x']
            self.tmp_people_y = people['y']

            self.tmp = self.tmp.data.copy()
            self.tmp['time'] = self.tmp['time'] + [self.pre_cal_step+1]
            self.tmp['Susceptible'] = self.tmp['Susceptible'] + [num_sus]
            self.tmp['Infectious'] = self.tmp['Infectious'] + [num_infec]
            self.tmp['Removed'] = self.tmp['Removed'] + [num_rem]
            self.tmp = ColumnDataSource(self.tmp)
            self.data_source.append(self.tmp.data)
            self.percent = f"{self.pre_cal_step/self.vars['total_steps']*100.:.2f}%"
            self._update_percentage()
            self.pre_cal_step += 1
            self.callback_id2 = curdoc().add_next_tick_callback(self._pre_cal_core)

    def _run(self):
        if self.step_index > self.run_step_limit:
            self._stop_period()
        else:
            try:
                self.people.data = self.people_source[self.step_index]
                self.data.data = self.data_source[self.step_index]
            except Exception as e:
                print(e)
                raise SystemError
            self.step_index += 1

    def _display(self):
        self.layout.children = [
            column(
                self.desc,
                row(
                    column(row(self.button_start, self.button_reset, width=200),
                           row(self.button_stop, self.button_replay, width=200),
                           self.slider_pop, self.silder_step_size, self.slider_radius, self.slider_infect_rate,
                           self.slider_total, self.slider_removed
                           ),
                    Div(width=50),
                    self.line_fig,
                    Div(width=50),
                    self.point_fig,
                    Div(width=50),
                    self.legend_fig,
                    width=1200
                )
            )
            # self.layout2[0]
        ]

    def _update(self):
        self._pre_calculate()

    def _random_walk(self):
        return np.random.normal(self.vars['step_size'], .8, self.vars['pop_size']) \
               * np.random.choice([1, -1], size=1)[0]

    def _update_people_pos(self, people_x=None, people_y=None):
        if people_x is None:
            people_x = self.people.data['x']
        if people_y is None:
            people_y = self.people.data['y']

        people = dict()
        tmp = people_x + self._random_walk()
        while np.sum(tmp > self.box_size) > 1 or np.sum(tmp < 0) > 1:
            # print('Try another value x')
            tmp = people_x + self._random_walk()
        people['x'] = tmp

        tmp = people_y + self._random_walk()
        while np.sum(tmp > self.box_size) > 1 or np.sum(tmp < 0) > 1:
            # print('Try another value y')
            tmp = people_y + self._random_walk()
        people['y'] = tmp
        return people

    def _cal_data(self, people, people_df=None):
        if people_df is None:
            people_df = pd.DataFrame(self.people.data)
        people_df['x'] = people['x']
        people_df['y'] = people['y']
        infected_id = people_df.loc[people_df['color'] == self.infect_color, 'ID'].values

        for id_ in infected_id:
            coords = people_df.loc[people_df['ID'] == id_, ['x', 'y']].values[0]
            people_df['dis'] = self._distance(np.array([coords[0]]*len(people_df)),
                                              np.array([coords[1]]*len(people_df)),
                                              people_df['x'].values, people_df['y'].values)

            people_df['may_infected'] = people_df['dis'] < self.vars['infection_radius']
            # print(people_df[people_df['may_infected'] == True])

            tmp = people_df[(people_df['may_infected'] == True) & (people_df['color'] == self.sus_color)]
            if len(tmp) > 0:
                test = np.random.uniform(low=0, high=100, size=len(tmp))
                infected = [self.infect_color if x <= self.vars['infection_rate'] * 100 else self.sus_color for x in test]
                tmp['color'] = infected
                tmp = tmp[tmp['color'] == self.infect_color]
                people_df.loc[tmp.index, 'color'] = [self.infect_color] * len(tmp)
            people_df = people_df.drop(['dis', 'may_infected'], axis=1)
        infected = people_df[people_df['color'] == self.infect_color]
        people_df.loc[infected.index, 'infect_days'] = people_df['infect_days'] + 1

        removed = people_df[people_df['infect_days'] > self.vars['removed_steps']]
        if len(removed) > 0:
            people_df.loc[removed.index, 'infect_days'] = [-1] * len(removed)
            people_df.loc[removed.index, 'color'] = [self.removed_color] * len(removed)

        return len(people_df[people_df['color'] == self.sus_color]), \
               len(people_df[people_df['color'] == self.infect_color]), \
               len(people_df[people_df['color'] == self.removed_color]), \
               people_df

    @staticmethod
    def _distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _update_percentage(self):
        self.text_s.data = dict(
            x=[50],
            y=[50],
            text=['processing {}'.format(self.percent)])

    def _show_loading_page(self):
        loading_page = figure(
            width=1000,
            height=200,
        )
        label = LabelSet(x='x', y='y',
                         text='text', source=self.text_s, x_offset=-25, y_offset=-25, text_font_size="25pt")
        loading_page.triangle(x='x', y='y', size=1, source=self.text_s, color='#2F2F2F')
        loading_page.add_layout(label)
        loading_page.xgrid.grid_line_color = None
        loading_page.ygrid.grid_line_color = None
        loading_page.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        loading_page.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        loading_page.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        loading_page.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        loading_page.xaxis.major_label_text_color = None  # turn off x-axis tick labels leaving space
        loading_page.yaxis.major_label_text_color = None  # turn off y-axis tick labels leaving space
        loading_page.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
        loading_page.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
        loading_page.outline_line_color = None
        loading_page.axis.visible = None
        loading_page.toolbar.logo = None
        loading_page.toolbar_location = None
        self.layout.children = [loading_page, row(Div(width=550), self.button_stop_run, Div(width=450))]

    def _get_legend(self):
        text_legend = ColumnDataSource(data=dict(
            x=[10, 10, 10],
            y=[70, 80, 90],
            text=['Removed', 'Infected ', 'Susceptible'],
            color=[self.removed_color, self.infect_color, self.sus_color],
            text_color=['#D3D3D3', '#D3D3D3', '#D3D3D3']
        ))
        self.legend_fig = figure(
            width=120,
            height=400,
            x_range=(0, 100), y_range=(0, 100)
        )
        label = LabelSet(x='x', y='y',
                         text='text', source=text_legend, x_offset=10, y_offset=-5, text_font_size="10pt", text_color='text_color')
        self.legend_fig.square(x='x', y='y', size=10, source=text_legend, color='color')
        self.legend_fig.add_layout(label)
        self.legend_fig.xgrid.grid_line_color = None
        self.legend_fig.ygrid.grid_line_color = None
        self.legend_fig.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        self.legend_fig.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        self.legend_fig.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        self.legend_fig.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        self.legend_fig.xaxis.major_label_text_color = None  # turn off x-axis tick labels leaving space
        self.legend_fig.yaxis.major_label_text_color = None  # turn off y-axis tick labels leaving space
        self.legend_fig.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
        self.legend_fig.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
        self.legend_fig.outline_line_color = None
        self.legend_fig.axis.visible = None
        self.legend_fig.toolbar.logo = None
        self.legend_fig.toolbar_location = None

    def _period_update(self):
        if self.start_flag:
            self.step_index = 0
            self.pre_cal_step = 0
            self.percent = "0%"
            self.people_source, self.data_source = [], []
            self.tmp_people_x = None
            self.tmp_people_y = None
            self.tmp_people_df = None
            self.tmp = None
            self.data.data = self._create_data().data
        else:
            self.start_flag = True
        self._show_loading_page()
        self.callback_id = curdoc().add_next_tick_callback(self._update)

    def _stop_period(self):
        curdoc().remove_periodic_callback(self.callback_id)

    def _stop_next(self):
        curdoc().remove_next_tick_callback(self.callback_id2)

    def _reset(self):
        curdoc().clear()
        self.__init__()

    def _replay(self):
        self.step_index = 0
        self.callback_id = curdoc().add_periodic_callback(self._run, 50)

    def _stop_run(self):
        self.early = True
        self.early_step = self.pre_cal_step
        self.pre_cal_step = self.vars['total_steps'] + 1
        # print(self.run_step_limit, self.pre_cal_step)

    def _get_plot(self):
        curdoc().title = "Epidemic Simulation"
        self.desc = Div(text=open('myapp/description.html').read(), sizing_mode="stretch_width")
        self.layout = column(
            self.desc,
            row(
                column(row(self.button_start, self.button_reset, width=200),
                       row(self.button_stop, self.button_replay, width=200),
                       self.slider_pop, self.silder_step_size, self.slider_radius, self.slider_infect_rate,
                       self.slider_total, self.slider_removed
                       ),
                Div(width=50),
                self.line_fig,
                Div(width=50),
                self.point_fig,
                Div(width=50),
                self.legend_fig,
                width=1200
            )
        )
        curdoc().add_root(self.layout)

# python -m bokeh serve virus_simulation --dev virus_simulation