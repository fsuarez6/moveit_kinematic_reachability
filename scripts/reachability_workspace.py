#!/usr/bin/env python

import rospy
import sys, subprocess, argparse, signal, os
import numpy as np
# Libraries to read openrave database
import h5py
import numpy
# Graphical things
import matplotlib.pyplot as plot

class ReachabilityWorkspace:
  def __init__(self, args):
    if args.show_profile:
      self.show_reachability_profile(args)
    if args.create_graph:
      self.create_reachability_graph(args)
    if args.find_database:
      self.find_database_files(args.robot_xml, args.manipname)

  def get_filename(self, robot_xml, manipname=None, database='kinematicreachability'):
    if manipname:
      filename = subprocess.check_output(['openrave.py','--database',str(database),'--robot=' + str(robot_xml),'--manipname=' + str(manipname), '--getfilename']).rstrip()
    else:
      filename = subprocess.check_output(['openrave.py','--database',str(database),'--robot=' + str(robot_xml), '--getfilename']).rstrip()
    return filename

  def show_reachability_profile(self, args):
    robot_xml = args.robot_xml
    scale = float(args.show_scale)
    thresholds = args.thresholds
    colourvals = args.colourvals
    draw_opaque = args.draw_opaque
    clip_visual_quarter = args.clip_visual_quarter
    clip_visual_half = args.clip_visual_half
    try:
      mlab = __import__('enthought.mayavi.mlab',fromlist=['mlab'])
    except ImportError:
      mlab = __import__('mayavi.mlab',fromlist=['mlab'])

    #fill in
    ppfile = self.get_filename(robot_xml, args.manipname)
    rospy.loginfo('Showing kinematic reachability for robot model: %s \n from file %s' % (robot_xml, ppfile))
    
    f = mlab.figure(1,fgcolor=(0,0,0), bgcolor=(1,1,1),size=(1024,768))
    mlab.clf()
    
    #load the file
    f=h5py.File(ppfile,'r')
    
    #get the reachability data out
    opacity=None
    reachability3d = numpy.minimum(f['reachability3d'].value*scale,1.0)
    print "Min Reachability : " + str(numpy.amin(reachability3d)) +  " Max Reachability : " + str(numpy.amax(reachability3d))
    print "Average Reachability : " + str(numpy.average(reachability3d))
    rospy.loginfo('max reachability: %r',numpy.amax(reachability3d))
    
    reachability3d[0,0,0] = 1 # have at least one point be at the maximum
    
    w,b,h = reachability3d.shape
    
    #threshold out the reachability
    if not thresholds == None:
      thresholds = map(float,args.thresholds)
      colourvals = map(float,args.colourvals)
      thresholded = reachability3d
      #apply lowest threshold
      mask = (reachability3d < thresholds[0])
      thresholded[mask] = 0 # seems to be necessary to cut off all under thresholds
      for i in range(1,len(thresholds)):
        mask = numpy.logical_and(reachability3d >= thresholds[i-1],reachability3d < thresholds[i])
        thresholded[mask] = colourvals[i-1]
      mask = reachability3d >= thresholds[-1]
      thresholded[mask] = 1
      reachability3d = thresholded
      contours=colourvals
      #~ opacity = colourvals
    else:
      contours=[0.01,0.1,0.2,0.5,0.8,0.9,0.99]
    
    #cut out a section for visualisation
    if clip_visual_quarter:
      reachability3d[w/2:-1, b/2:-1, h/2:-1] = 0
    if clip_visual_half:
      reachability3d[w/2:-1,:,:] = 0
     
    # Start the pipeline
    offset = numpy.array((0,0,0))
    src = mlab.pipeline.scalar_field(reachability3d)
    
    if draw_opaque:
      opacity=numpy.ones(len(contours))
    for i,c in enumerate(contours):
      mlab.pipeline.iso_surface(src,contours=[c],opacity=min(1,0.7*c if opacity is None else opacity[i]))

    # Plot zero point
    mlab.pipeline.scalar_scatter(0,0,0,3,color=(0,0,0))
    centre_src=mlab.pipeline.scalar_scatter([0],[0],[0],[2],color=(0,0,0))
    mlab.pipeline.glyph(centre_src, scale_mode='none', scale_factor=1)
    statpts = f['reachabilitystats'].value[0::100,0:3]
    point_src = mlab.points3d(statpts[:,0],statpts[:,1],statpts[:,2],[0]*len(statpts[:,0]))
    mlab.pipeline.glyph(point_src, scale_mode='none', scale_factor=0.1)
    
    mlab.show()

  def create_reachability_graph(self, args):
    robot_xml = args.robot_xml
    n_data_points = int(args.data_points)
    show_graph = args.show_graph
    log_file = args.log_file
    clip_zero_pts = args.clip_zero_pts
    volume_graph = args.volume_graph

    from scipy.spatial import ConvexHull
    from scipy.spatial import Delaunay
    from mpl_toolkits.mplot3d import Axes3D

    rospy.loginfo('Loading reachability database for %s' % robot_xml)

    # load up the data
    ppfile = self.get_filename(robot_xml, args.manipname)
    f=h5py.File(ppfile,'r')
    reachability3d = f['reachability3d'].value
    
    total_pts = reachability3d.size
    
    nonzero_tot_pts = (reachability3d > 0).sum()
    rospy.loginfo('%d out of %d total tested points are reachable (%f%%)' % (nonzero_tot_pts,total_pts,(float(nonzero_tot_pts)/float(total_pts))*100.))
    
    #~ if volume_graph:
      #~ from scipy import ndimage #for volume graph
      #~ desired_reach = 0.95
      #~ volmask = reachability3d >= desired_reach
      #~ regions, nregs = ndimage.label(volmask)
      #~ 
      #~ fig = plot.figure()
      #~ ax = fig.add_subplot(111, projection='3d')
      #~ x,y,z = numpy.where(reachability3d >= desired_reach)
      #~ every = 2
      #~ ax.scatter(x[0::every],y[0::every],z[0::every],c=((0.5,0.5,1,0.8)))
      #~ 
      #~ cmap = plot.get_cmap('jet')
      #~ for reg in range(1,nregs):
        #~ x,y,z = numpy.where(regions == reg)
        #~ print "region %s: %s points"%(reg,x.size)
        #~ ax.scatter(x[0::every],y[0::every],z[0::every],c=cmap(float(reg)/float(nregs)))
        #~ 
      #~ plot.show()
      #~ exit()
    
    #orientations range between 0 and 1
    #~ orientations = [float(x)/float(n_data_points) + 1/float(n_data_points) for x in range(n_data_points)]
    orientations = [float(x)/float(n_data_points) for x in range(n_data_points)]
    
    #omit points with zero reachability (improves graph)
    if clip_zero_pts:
      orientations[0] += 1/(100*float(n_data_points))
    
    rospy.loginfo('Calculating Reachable Zone Convex Hull')
    
    #get nonzero indices
    x,y,z = numpy.where(reachability3d > 0)
    nz_points = numpy.array([[x[i],y[i],z[i]] for i in range(len(x))])
    convex_hull = ConvexHull(nz_points)
    
    #Plotting! plot the convex hull
    #~ fig = plot.figure()
    #~ ax = fig.add_subplot(111, projection='3d')
    #~ colourmap = []
    #~ colours = plot.get_cmap('jet')
    #~ for i in range(len(x)):
      #~ pt_val = reachability3d[x[i],y[i],z[i]]
      #~ col = colours(pt_val/1.2)
      #~ colour = col#(col[0],col[1],col[2],pt_val) #vary alpha too
      #~ colourmap.append(colour)
    #~ every = 5
    #~ ax.scatter(x[0::every],y[0::every],z[0::every],c=colourmap[0::every],marker='o')
    #~ for simplex in convex_hull.simplices:
      #~ plot.plot(x[simplex],y[simplex],z[simplex],'--',c=(1,0,0,1))
    #~ plot.show()
    
    #~ #Plotting! plot 2D convex hull
    #~ fig = plot.figure()
    #~ colourmap = []
    #~ colours = plot.get_cmap('jet')
    #~ for i in range(len(x)):
      #~ if z[i] > 65 and z[i] < 75:
        #~ pt_val = reachability3d[x[i],y[i],z[i]]
        #~ col = colours(pt_val*5)
        #~ plot.plot(y[i],x[i],'o',c=col)
    #~ plot.show() 
    #~ fig = plot.figure()
    #~ colourmap = []
    #~ colours = plot.get_cmap('jet')
    #~ for i in range(len(x)):
      #~ if x[i] > 50 and x[i] < 60:
        #~ pt_val = reachability3d[x[i],y[i],z[i]]
        #~ col = colours(pt_val*5)
        #~ plot.plot(y[i],z[i],'o',c=col)
    #~ plot.show() 
    
    #~ #Plotting! plot whole workspace
    #~ fig = plot.figure()
    #~ ax = fig.add_subplot(111, projection='3d')
    #~ colourmap = []
    #~ colours = plot.get_cmap('jet')
    #~ for i in range(len(x)):
      #~ pt_val = reachability3d[x[i],y[i],z[i]]
      #~ col = colours(pt_val)
      #~ colourmap.append(col)
    #~ every = 10
    #~ ax.scatter(x[0::every],y[0::every],z[0::every],c=colourmap[0::every])
    
    
    #get all indices -note to self: replace for something more elegant
    #than just finding all points with a value over minus 1 (i.e. all points)
    x,y,z = numpy.where(reachability3d > -1)
    all_points = numpy.array([[x[i],y[i],z[i]] for i in range(len(x))])
    
    #find all the points inside the convex hull
    inhull = Delaunay(convex_hull.simplices).find_simplex(all_points)>=0
    
    #~ inhull2 = Delaunay(convex_hull.simplices).find_simplex(convex_hull.points)>=0
    #~ print convex_hull.points[0:10]
    #~ print "matchingpoints:%s"%(inhull2.sum())
    #~ print "lenpts:%s"%(len(convex_hull.points))
    
    #~ #Plotting! plot whole workspace
    #~ every = 200
    #~ ax.scatter(x[0::every],y[0::every],z[0::every],c=((0.5,0.5,1,0.2)))
    #~ plot.show()
    #~ print inhull2.shape
    #~ print all_points.shape
    #Plotting! plot only points in the convex hull
    #~ fig = plot.figure()
    #~ ax = fig.add_subplot(111, projection='3d')
    #~ foundpts = all_points[inhull2]
    #~ print foundpts.shape
    #~ every = 50
    #~ ax.scatter(foundpts[0::every,0],foundpts[0::every,1],foundpts[0::every,2],c='b',marker='o')
    #~ for simplex in convex_hull.simplices:
      #~ plot.plot(convex_hull.points[simplex,0],convex_hull.points[simplex,1],convex_hull.points[simplex,2],'--',c=(1,0,0,1))
    #~ plot.show()
    
    #count total points inside hull
    pts_in_hull = inhull.sum()

    #make a mask of only points in hull
    mask = reachability3d < -1 #cludgy way to make a false boolean array of the right size
    for i in range(len(all_points)):
      mask[all_points[i][0],all_points[i][1],all_points[i][2]] = inhull[i]
    
    rospy.loginfo("Total Points: %d, Point in hull: %d, Percent in hull: %f%%"%(total_pts,pts_in_hull,float(pts_in_hull)/float(total_pts)*100.))
    
    rospy.loginfo("Creating Reachability Profile")
    
    percent_in_group = []
    for level in orientations:
      #count the number of points with this percentage of orientations
      n_points =  (reachability3d[mask] >= level).sum()
      percent_in_group.append(float(n_points)/float(pts_in_hull))
    
    if not clip_zero_pts:
      #set first point of profile only to those points with zero reachability
      #(otherwise this value will always be 1 - i.e. all points have 
      #at least 0 reachability, which tells us nothing)
      n_points =(reachability3d[mask] == 0.0).sum()
      percent_in_group[0] = float(n_points)/float(pts_in_hull)
      rospy.loginfo("%d points with zero reachability (%f%% of the points in hull)"%(n_points, percent_in_group[0]*100.))
      
    if volume_graph:
      from scipy import ndimage #for volume calculation
      samplevol = f['xyzdelta'].value**3 #volume of each sample
      #generate the reachability profile in volume
      volume_in_group = []
      for level in orientations:
        #make binary image of the relevant points
        volmask = reachability3d >= level
        
        #calculate the "connected components"
        regions, nregs = ndimage.label(volmask)
        
        #get the biggest connected region
        region_points = [(regions==reg).sum() for reg in range(1,nregs+1)]
        if (len(region_points) > 0):
          volume_in_group.append(max(region_points)*samplevol)
        else:
          volume_in_group.append(0)

    if show_graph:
      rospy.loginfo("Plotting Reachability Profile")
      fig = plot.figure()
      if volume_graph:
        plot.plot(orientations, volume_in_group, '-', linewidth=2)
        plot.ylabel('Max connected volume (m^3)')
        
        #allows clicking on the graph to generate plot of volume
        def onclick(event):
          print event.x, event.xdata
          volmask = reachability3d >= event.xdata
          regions, nregs = ndimage.label(volmask)
          region_points = [(regions==reg).sum() for reg in range(1,nregs+1)]
          regidx = region_points.index(max(region_points))
          x,y,z = numpy.where(regions == regidx+1)
          print regidx
          fig2 = plot.figure()
          ax = fig2.add_subplot(111, projection='3d')
          for simplex in convex_hull.simplices:
            plot.plot(convex_hull.points[simplex,0],convex_hull.points[simplex,1],convex_hull.points[simplex,2],'--',c=(0.5,0.5,1,1))
          #~ every = 10
          #~ ax.scatter(x[0::every],y[0::every],z[0::every],c=(1,0,0,1),marker='o')
          vol_points = numpy.array([[x[i],y[i],z[i]] for i in range(len(x))])
          vol_hull = ConvexHull(vol_points)
          from mpl_toolkits.mplot3d.art3d import Poly3DCollection
          
          for simplex in vol_hull.simplices:
            verts = [zip(vol_hull.points[simplex,0],vol_hull.points[simplex,1],vol_hull.points[simplex,2])]
            ax.add_collection3d(Poly3DCollection(verts))
            #~ plot.plot(vol_hull.points[simplex,0],vol_hull.points[simplex,1],vol_hull.points[simplex,2],'--',c=(1,0,0,1))
          plot.title("Reachability >= %s"%(event.xdata,))
          plot.show()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
      else:
        plot.plot(orientations, percent_in_group, '-', linewidth=2)
        plot.ylabel('Fraction Points')
      plot.xlabel('Reachability')
      plot.show()

  def find_database_files(self, robot_xml, manipname):
    rospy.loginfo('Finding location of database files for %s' % robot_xml)
    ppfile = self.get_filename(robot_xml, manipname)
    kinematicsfile = self.get_filename(robot_xml, manipname, 'inversekinematics')

    rospy.loginfo('Reachability Location: %s' % ppfile)
    if os.path.isfile(ppfile):
      rospy.loginfo('Reachability database exists')
    else:
      rospy.logwarn('Reachability database Does Not Exist')

    rospy.loginfo('Inverse Kinematics Location: %s' % kinematicsfile)
    if os.path.isfile(kinematicsfile):
      rospy.loginfo('Inverse Kinematics database exists')
    else:
      rospy.logwarn('Kinematics database Does Not Exist')



if __name__ == '__main__':
  node_name = 'kinematic_reachability_workspace'
  rospy.init_node(node_name)
  
  #parse the commandline arguments
  parser = argparse.ArgumentParser(description='Reachability Workspace', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  #mode flags
  parser.add_argument('-s','--show-profile',help='Show the reachability profile',action='store_true')
  parser.add_argument('-c','--create-graph',help='Generate the reachability profile graph',action='store_true')
  parser.add_argument('-f','--find-database',help='Finds the database files for the supplied robot',action='store_true')
  
  #mode specific arguments
  #mode: --create-robot
  parser.add_argument('-r', '--robot-xml', help='path to the *.robot.xml file previously used by openrave to create kinematic reachability', type=str)
  parser.add_argument('-m', '--manipname', help='The name of the manipulator on the robot to use', type=str)
  
  #mode: --show-profile
  parser.add_argument('-ss', '--show-scale', help='to scale colours for visualisation using --show-profile', default=1)
  parser.add_argument('-st', '--thresholds', help='multiple thresholds for thresholding visualisation', nargs='+', default=None)
  parser.add_argument('-sc', '--colourvals', help='colour values for the thresholds', nargs='+', default=None)
  parser.add_argument('-so', '--draw-opaque', help='draw completely opaque, i.e. no transparency grading', action='store_true')
  parser.add_argument('-svq', '--clip-visual-quarter', help='cuts out a quarter section of the visualization for easier viewing', action='store_true')
  parser.add_argument('-svh', '--clip-visual-half', help='cuts out a half section of the visualization for easier viewing', action='store_true')
  
  #mode: --create-graph
  parser.add_argument('-cp', '--data-points', help='the number of data points in the graph', default=100)
  parser.add_argument('-cs', '--show-graph', help='display the graph in python', action='store_true')
  parser.add_argument('-cv', '--volume-graph', help='plots reachability by volume instead of percent of points', action='store_true')
  parser.add_argument('-cl', '--log-file', help='logs the generated graph to a file', default=None)
  parser.add_argument('-cz', '--clip-zero-pts', help='Does not include points with zero reachability', action='store_true')
  
  # Hack args to work with roslaunch
  myargs = rospy.myargv(sys.argv)
  myargs = myargs[1:]
  # Parse args
  args = parser.parse_args(myargs)

  # Error check
  if not (args.robot_xml):
    parser.error('--robot-xml must be provided')
  if args.thresholds or args.colourvals:
    if not (args.thresholds and args.colourvals):
      parser.error('--thresholds and --colourvals must be provided together')
    elif len(args.thresholds) != len(args.colourvals):
      parser.error('--colourvals and --thresholds must be the same length')
    elif not args.show_profile:
        parser.error('--thresholds and --colourvals are used with --show-profile')

  ReachabilityWorkspace(args)
