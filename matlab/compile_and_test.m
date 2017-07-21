clear
close all
%% compile
LIBIGL_DIR='/Users/olkido/Dropbox/Work/code/other/libigl/';
matlab_command = sprintf('mex -I%s/include -I%s/external/nanogui/ext/eigen  cut_mesh_mex.cpp',LIBIGL_DIR,LIBIGL_DIR);
eval(matlab_command)

%% test
[V,F] = readOFF('../data/sphere_0.off');
[Vc,Fc] = cut_mesh_mex(V,F);

% plot and highlight boundary in cut (and original) mesh
[E] = exterior_edges(F);
[Ec] = exterior_edges(Fc);

figure(1);clf
% original mesh and its boundary
subplot(1,2,1);
patch('Faces', F, 'Vertices', V, 'FaceColor', [1,1,1], 'EdgeColor', [.7,.7,.7]);
hold on;
s = V(E(:,1),:); e = V(E(:,2),:);
line([s(:,1)';e(:,1)'], [s(:,2)';e(:,2)'], [s(:,3)';e(:,3)'], 'Color', 'r', 'LineWidth', 3);
hold off;
axis equal; axis tight; axis off; 

% cut mesh and its boundary (might need to rotate to see it)
subplot(1,2,2);
patch('Faces', Fc, 'Vertices', Vc, 'FaceColor', [1,1,1], 'EdgeColor', [.7,.7,.7]);
hold on;
s = Vc(Ec(:,1),:); e = Vc(Ec(:,2),:);
line([s(:,1)';e(:,1)'], [s(:,2)';e(:,2)'], [s(:,3)';e(:,3)'], 'Color', 'r', 'LineWidth', 3);
hold off;
axis equal; axis tight; axis off


