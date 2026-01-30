class Single_Cylinder {
  BDIM flow;
  Body body;
  boolean QUICK = true, order2 = true;
  int n, m, out, up, resolution,NT=1;
  float dt, t, D, xi0, xi1, xi2, xi3, theta;//initial dt=1 only determine BDIM's update u.
  float xi0_m, xi1_m, xi2_m, xi3_m, gR, theta_m, r, dphi0, dphi1, dphi2, dphi3;
  FloodPlot flood;
  PVector force, force_0, force_1, force_2;
  ArrayList<PVector> xc=new ArrayList<PVector>();
  float D_0, D_1;

  Single_Cylinder (int resolution, int Re,  PVector xc_1, float dia_0, float xi0, float dtReal, int xLengths, int yLengths, boolean isResume) {
    n = xLengths*resolution;
    m = yLengths*resolution;
    this.resolution = resolution;
    this.xi0 = xi0;
    xi0_m=xi0;

    // println(D_0 + " " + dia_0);

    this.dt = dtReal*this.resolution;

    Window view = new Window(0, 0, n, m); // zoom the display around the body
    D=resolution;

    D_0 = dia_0*D;
    
    body = new CircleBody(xc_1.x, xc_1.y, dia_0*D, view);
  
    flow = new BDIM(n,m,dt,body,(float)D/Re,QUICK);
    
    if(isResume){
      flow.resume("saved\\init\\init.bdim");
    }
    
    flood = new FloodPlot(view);
    flood.range = new Scale(-1, 1);
    flood.setLegend("vorticity"); 
  }


 void update2(){
   flow.dt = dt;
   dphi0 = (2*xi0_m*dt)/D_0;

   body.rotate(dphi0);
                 
   flow.update(body);
   if (order2) {flow.update2(body);}
   println("t="+nfs(t,2,2)+";  ");
   t += dt/resolution;  //nonedimension
   force_0 = body.pressForce(flow.p).mult(-1);
 }

 void update(){
    for ( int i=0 ; i<NT ; i++ ) {
      if (flow.QUICK) {
        dt = flow.checkCFL();
        flow.dt = dt;
      }
      dphi0 = (2*xi0_m*dt)/D; //anglar velocity
      body.rotate(dphi0);//change index try;
              
       flow.update(body);
       if (order2) {flow.update2(body);}
       print("t="+nfs(t,2,2)+";  ");
       t += dt/resolution;  //nonedimension
       force = body.pressForce(flow.p).mult(-1);
  }
}

  void display(float xPos, float yPos, float targetx, float targety) {
    flood.display(flow.u.curl());
    body.display();
    flood.displayTime(t);

  }
}
