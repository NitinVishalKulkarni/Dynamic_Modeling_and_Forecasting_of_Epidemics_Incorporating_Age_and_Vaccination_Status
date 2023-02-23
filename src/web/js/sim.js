class SimUI {
    constructor(divoc) {
        this.divoc = divoc

        this.dim = {
            w: 1000,
            h: 500,
            u: 20,
        }

        this.canvas = document.getElementById('canvas')
        this.canvas.width = this.dim.w
        this.canvas.height = this.dim.h

        paper.setup(canvas)
        this.draw()
    }

    draw() {
        paper.project.clear()
        let d = this.dim
        let s = this.divoc.state

        let ch = d.h / 2
        let u = d.u

        this.compartment(u * 7, ch, s.populations.susceptible, 'Susceptible')
        this.compartment(u * 15, ch + (u * 5), s.populations.exposed1, 'exposed1')
        this.compartment(u * 15, ch - (u * 5), s.populations.exposed2, 'exposed2')
        this.compartment(u * 23, ch, s.populations.infected, 'infected')
        this.compartment(u * 36, ch - (u * 5), s.populations.recovered, 'recovered')
        this.compartment(u * 36, ch, s.populations.hospitalized, 'hospitalized')
        this.compartment(u * 36, ch + (u * 5), s.populations.deceased, 'deceased')
    }

    // ----------------------------------------

    compartment(cx, cy, population, name){
        let u = this.dim.u
        let w = 9
        this.rect(cx, cy, u, u * w, 'black', name + ': ' + (population.uv + population.fv + population.b))
        this.rect(cx - (u * w/3), cy + u, u, u * w/3, 'black', 'uv: ' + population.uv)
        this.rect(cx, cy + u, u, u * w/3, 'black', 'fv: ' + population.fv)
        this.rect(cx + (u * w/3), cy + u, u, u * w/3, 'black', 'b: ' + population.b)
    }

    circle(cx, cy, r, strokeColor='black', fillColor=null) {
        let circle = new paper.Path.Circle(new paper.Point(cx, cy), r);
        circle.strokeColor = strokeColor
        if (fillColor != null)
            circle.fillColor = fillColor
        return circle
    }

    rect(cx, cy, b, l, strokeColor='black', text=null) {
        let rect = new paper.Path.Rectangle(cx-l/2, cy-b/2, l, b)
        rect.strokeColor = strokeColor
        if (text != null)
            this.text(cx, cy, text)
    }

    text(cx, cy, content, fillColor='black') {
        let text = new paper.PointText(new paper.Point(cx, cy + 3))
        text.justification = 'center'
        text.fillColor = fillColor
        text.content = content
    }
}