class Move_ball {
  BDIM flow;
  BodyUnion body;
  boolean QUICK = true, order2 = true;
  int n, m, out, up, resolution,NT=1;
  float dt, t, D, xi0, xi1, xi2, xi3, theta;//initial dt=1 only determine BDIM's update u.
  float xi0_m, xi1_m, xi2_m, xi3_m, gR, theta_m, r, dphi0, dphi1, dphi2, dphi3;
  FloodPlot flood;
  PVector force, force_0, force_1, force_2;
  //PVector xc_1= new PVector(), xc_2= new PVector(), xc_3= new PVector();
  ArrayList<PVector> xc=new ArrayList<PVector>();

  Move_ball (int resolution, int Re,  PVector xc_1, PVector xc_2, PVector xc_3, PVector xc_4,  float xi0,  float xi1,  float xi2, float xi3, float dtReal, int xLengths, int yLengths, boolean isResume) {
    n = xLengths*resolution;
    m = yLengths*resolution;
    this.resolution = resolution;
    this.xi0 = xi0;
    this.xi1 = xi1;
    this.xi2 = xi2;
    this.xi3 = xi3;
    xi0_m=5*xi0;
    xi1_m=5*xi1;
    xi2_m=5*xi2;
    xi3_m=5*xi3;
    this.dt = dtReal*this.resolution;

    Window view = new Window(0, 0, n, m); // zoom the display around the body
    D=resolution;
    
    xc.add( xc_1 );
    xc.add( xc_2 );
    xc.add( xc_3 );
    xc.add( xc_4 );
    
    body =new BodyUnion(new EllipseBody(xc_1.x, xc_1.y, D/4, 0.20, view),
    new EllipseBody(xc_2.x, xc_2.y, D/4, 0.20, view),
    new EllipseBody(xc_3.x, xc_3.y, D/4, 0.20, view),
    new EllipseBody(xc_4.x, xc_4.y, D/4, 0.20, view));
    flow = new BDIM(n,m,dt,body,(float)D/Re,QUICK);
    
    if(isResume){
      flow.resume("saved\\init\\init.bdim");
    }
    
    flood = new FloodPlot(view);
    flood.range = new Scale(-1, 1);
    flood.setLegend("vorticity"); 
  }


 void update2(){
   //dt = flow.checkCFL();
   flow.dt = dt;
   dphi0 = (2*xi0_m*dt)/D; // xi0_m->dphi0 线速度到角速度(弧度）
   dphi1 = (2*xi1_m*dt)/D;
   dphi2 = (2*xi2_m*dt)/D;
   dphi3 = (2*xi3_m*dt)/D;
   body.bodyList.get(0).rotate(dphi0);//change index try;
   body.bodyList.get(1).rotate(dphi1);//change index try;
   body.bodyList.get(2).rotate(dphi2);
   body.bodyList.get(3).rotate(dphi3);
   
   //print("cccccccccccccccccccccccc"+body.xc+";");
              
   flow.update(body);
   if (order2) {flow.update2(body);}
   println("t="+nfs(t,2,2)+";  ");
   t += dt/resolution;  //nonedimension
  
   force = body.bodyList.get(0).pressForce(flow.p).mult(-1);
   //force_0 = body.bodyList.get(0).pressForce(flow.p).mult(-1);  //multply calculation to -1
   force_1 = body.bodyList.get(1).pressForce(flow.p).mult(-1);
   force_2 = body.bodyList.get(2).pressForce(flow.p).mult(-1);
   force.add(force_1);
   force.add(force_2);
 }

 void update(){
    for ( int i=0 ; i<NT ; i++ ) {
      if (flow.QUICK) {
        dt = flow.checkCFL();
        flow.dt = dt;
      }
      dphi0 = (2*xi0_m*dt)/D; //anglar velocity
      dphi1 = (2*xi1_m*dt)/D;
      dphi2 = (2*xi2_m*dt)/D;
      dphi3 = (2*xi3_m*dt)/D;
      body.bodyList.get(0).rotate(dphi0);//change index try;
      body.bodyList.get(1).rotate(dphi1);//change index try;
      body.bodyList.get(2).rotate(dphi2);
      body.bodyList.get(3).rotate(dphi3);
              
       flow.update(body);
       if (order2) {flow.update2(body);}
       print("t="+nfs(t,2,2)+";  ");
       t += dt/resolution;  //nonedimension
       force = body.bodyList.get(0).pressForce(flow.p).mult(-1);
       //force_0 = body.bodyList.get(0).pressForce(flow.p).mult(-1);  //multply calculation to -1
       force_1 = body.bodyList.get(1).pressForce(flow.p).mult(-1);
       force_2 = body.bodyList.get(2).pressForce(flow.p).mult(-1);
       force.add(force_1);
       force.add(force_2); 
  }
}

  void display(float xPos, float yPos, float targetx, float targety) {
    flood.display(flow.u.curl());
    body.display();
    flood.displayTime(t);
    
    
    float xPos_target = 140; 
    float yPos_target = 55; 
    float diameter_target = 4; 
    float diameter = 3; 
    
    fill(255, 0, 0); 
    ellipse(targetx, targety, diameter_target, diameter_target); 
    fill(100, 255, 200);
    ellipse(xPos, yPos, diameter, diameter); 

  }
}
