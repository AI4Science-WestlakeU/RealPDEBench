class Test0{
    float dt=0.1,dtheta1,dtheta2;

    /////Ini_global
    float Re_test;//Re=6000
    float St_test;//Str=0.45
    int resolution_test; final int Chord_lengthbyY = 8;//gird resolution
    /////Ini_test
    float radius; //R of the cylinder
    float massC;  //mass of the cylinder
    float kC;     //spring K of the cylinder
    float betaC;  //beta of the cylinder
    float massR_C=18.2;//mass ratio
    float fn_C=1;   //initial freq
    float epi_C=0.8;  //cr/c
    /////Input_end
    float nu = resolution_test/Re_test;
    int n=resolution_test * Chord_lengthbyY;//all resolution in direction Y 
    /////Ini_endB
    BDIM flow;
    Body Cylinder1;
    Body Cylinder0;
    BodyUnion body;
    FloodPlot flood;
    Window window;
    float t=0;
    PVector force_0, force_1;

    Test0( int resolution, float radius, float massR_C, 
           float fn_C, float epi_C, float Re, float st, boolean QUICK) {
        Re_test = Re;
        this.radius=radius;
        this.massC=massR_C*PI/4.;
        this.kC=sq(fn_C)*massR_C*pow(PI,3);
        this.betaC=sq(PI)*massR_C*resolution*fn_C*epi_C;

        resolution_test=resolution;
        n=resolution_test * Chord_lengthbyY;
        window = new Window(n, n);
        nu = resolution_test/Re_test;
        t = 0;

        Cylinder1 = new CircleBody(212,n/2,resolution_test,window);
        Cylinder0 = new CircleBody(n/12,n/2,resolution_test,window);

        body = new BodyUnion(Cylinder0,Cylinder1);

        PVector xc0 = Cylinder1.xc;
        flow = new BDIM(n,n,0,body,(float)resolution_test/Re_test, QUICK);

        flood = new FloodPlot(window);
        flood.range = new Scale(-.5,.5);
        flood.setLegend("vorticity");
        flood.setColorMode(1);
         }


    void updatePosition(float dt){
        Cylinder1.react(forceR(Cylinder1, new PVector(n/4,n/2)), momentR(Cylinder1), dt);
    }
    void update() {
        flow.dt = dt;
        updatePosition(dt);
        flow.update(body);flow.update2();
        t += dt;
        println(t);
        force_0 = body.bodyList.get(0).pressForce(flow.p).mult(-1);
        force_1 = body.bodyList.get(1).pressForce(flow.p).mult(-1);
    }
    void display() {
        flood.display(flow.u.curl());
        body.display();
    }

    PVector forceR(Body cylinder, PVector xc0){
        float fxR = cylinder.pressForce(flow.p).mult(-1).x-1.*betaC*cylinder.dotxc.x-1*kC*(cylinder.xc.x-xc0.x); // betac drag
        float fyR = cylinder.pressForce(flow.p).mult(-1).y-1.*betaC*cylinder.dotxc.y-1*kC*(cylinder.xc.y-xc0.y);
        return new PVector(fxR,fyR);
    }
    float momentR(Body cylinder){
        return cylinder.pressMoment(flow.p)*(-1);
    }

}