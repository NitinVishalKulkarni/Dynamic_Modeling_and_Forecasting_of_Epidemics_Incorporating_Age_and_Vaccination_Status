class Stats {
    constructor(env) {
        this.env = env
        this.setupCharts()
    }

    setupCharts() {
        this.graphsRef = document.getElementById('graphs')

        this.populationChart = this.addChartToPage({
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'susceptible',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'blue',
                    borderWidth: 1,
                    pointRadius: 0,
                }, {
                    label: 'exposed1',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'yellow',
                    borderWidth: 1,
                    pointRadius: 0,
                }, {
                    label: 'exposed2',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'orange',
                    borderWidth: 1,
                    pointRadius: 0,
                }, {
                    label: 'infected',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'red',
                    borderWidth: 1,
                    pointRadius: 0,
                }, {
                    label: 'recovered',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'green',
                    borderWidth: 1,
                    pointRadius: 0,
                }, {
                    label: 'hospitalized',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'pink',
                    borderWidth: 1,
                    pointRadius: 0,
                }, {
                    label: 'deceased',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'gray',
                    borderWidth: 1,
                    pointRadius: 0,
                },
                ]
            },
            options: {
                title: {
                    display: true,
                    text: `Population`
                },
                elements: {
                    line: {
                        tension: 0
                    }
                },
                scales: {
                    yAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'Population'
                        }
                    }],
                    xAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'TimeSteps'
                        }
                    }],
                },
            }
        })
        this.subCompPopulationChart = this.addChartToPage({
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'uv',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'red',
                    borderWidth: 1,
                    pointRadius: 0,
                }, {
                    label: 'fv',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'green',
                    borderWidth: 1,
                    pointRadius: 0,
                }, {
                    label: 'b',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'blue',
                    borderWidth: 1,
                    pointRadius: 0,
                }]
            },
            options: {
                title: {
                    display: true,
                    text: `Population - Sub Compartments`
                },
                elements: {
                    line: {
                        tension: 0
                    }
                },
                scales: {
                    yAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'Population'
                        }
                    }],
                    xAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'TimeSteps'
                        }
                    }],
                },
            }
        })
        this.eppChart = this.addChartToPage({
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'epp',
                    data: [],
                    backgroundColor: 'transparent',
                    borderColor: 'blue',
                    borderWidth: 1,
                    pointRadius: 0,
                }]
            },
            options: {
                title: {
                    display: true,
                    text: `EPP`
                },
                elements: {
                    line: {
                        tension: 0
                    }
                },
                scales: {
                    yAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'EPP'
                        }
                    }],
                    xAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'TimeSteps'
                        }
                    }],
                },
            }
        })

        // // Params Chart
        // let probs = [
        //     'vfv',
        //     'vb',
        //     's_e1',
        //     'e1_s',
        //     'e1_i',
        //     'i_r',
        //     'i_h',
        //     'i_d',
        //     'e2_i',
        //     'h_r',
        //     'h_d',
        //     'r_e2',
        //     'e2_r',
        // ]
        // let colors = [
        //     '#ff0000',
        //     '#ff5500',
        //     '#ffff00',
        //     '#00ff00',
        //     '#0000ff',
        //     '#7ae2a0',
        //     '#394a31',
        //     '#efbf8d',
        //     '#b766d5',
        //     '#db428f',
        //     '#e78db5',
        //     '#9a00eb',
        //     '#261b89',
        // ]
        // let datasets = probs.map((prob, index) => {
        //     return {
        //         label: prob,
        //         data: [],
        //         backgroundColor: 'transparent',
        //         borderColor: colors[index],
        //         borderWidth: 1,
        //         pointRadius: 0,
        //     }
        // })
        // this.probsChart = this.addChartToPage({
        //     type: 'line',
        //     data: {
        //         labels: [],
        //         datasets: datasets
        //     },
        //     options: {
        //         title: {
        //             display: true,
        //             text: `Probs`
        //         },
        //         elements: {
        //             line: {
        //                 tension: 0
        //             }
        //         },
        //         scales: {
        //             yAxes: [{
        //                 scaleLabel: {
        //                     display: true,
        //                     labelString: 'Params'
        //                 }
        //             }],
        //             xAxes: [{
        //                 scaleLabel: {
        //                     display: true,
        //                     labelString: 'TimeSteps'
        //                 }
        //             }],
        //         },
        //     }
        // })
    }
    
    step() {
        this.updatePopulationChart()
        this.updateSubCompPopChart()
        this.updateEppChart()
        // this.updateProbsChart()
    }

    updatePopulationChart() {
        let compartments = [
            'susceptible',
            'exposed',
            'exposed',
            'infected',
            'recovered',
            'hospitalized',
            'deceased',
        ]
        compartments.forEach((comp, index) => {
            let population = this.env.state.populations[comp]
            let total = population.uv + population.fv + population.b
            this.populationChart.data.datasets[index].data.push(total)
        })
        this.populationChart.data.labels.push(this.env.state.step_count)
        this.populationChart.update(this.env.state.step_count)
    }

    updateSubCompPopChart() {
        let uv = 0
        let fv = 0
        let b = 0
        let compartments = [
            'susceptible',
            'exposed',
            'exposed',
            'infected',
            'recovered',
            'hospitalized',
            'deceased',
        ]
        compartments.forEach((comp, index) => {
            let population = this.env.state.populations[comp]
            uv += population.uv
            fv += population.fv
            b += population.b
        })
        this.subCompPopulationChart.data.datasets[0].data.push(uv)
        this.subCompPopulationChart.data.datasets[1].data.push(fv)
        this.subCompPopulationChart.data.datasets[2].data.push(b)
        this.subCompPopulationChart.data.labels.push(this.env.state.step_count);
        this.subCompPopulationChart.update(this.env.state.step_count)
    }

    updateEppChart() {
        this.eppChart.data.datasets[0].data.push(this.env.state.epp)
        this.eppChart.data.labels.push(this.env.state.step_count)
        this.eppChart.update(this.env.state.step_count)
    }

    updateProbsChart(){
        let probs = [
            'vfv',
            'vb',
            's_e1',
            'e1_s',
            'e1_i',
            'i_r',
            'i_h',
            'i_d',
            'e2_i',
            'h_r',
            'h_d',
            'r_e2',
            'e2_r',
        ]
        probs.forEach((prob, index) => {
            this.probsChart.data.datasets[index].data.push(this.env.state.probs[prob])
        })
        this.probsChart.data.labels.push(this.env.state.step_count)
        this.probsChart.update(this.env.state.step_count)
    }

    addChartToPage(chartConfig) {
        let chartDiv = document.createElement('div')
        chartDiv.className = 'col-auto'
        chartDiv.style.width = '600px'
        chartDiv.style.height = '300px'
        let chartCanvas = document.createElement('canvas')

        let chart = new Chart(chartCanvas, chartConfig)

        chartDiv.appendChild(chartCanvas)
        this.graphsRef.append(chartDiv)

        return chart
    }

}