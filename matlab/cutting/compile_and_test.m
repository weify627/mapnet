clear
close all
%% compile
LIBIGL_DIR='/Users/olkido/Dropbox/Work/code/other/libigl/';
% list cpp files: the main one (with the mexFunction) and any dependencies
% order here matters: put the "main" .cpp file (the one that determines the
% name of the mex file) first here.
cfiles = {...
    'cut_mesh_mex.cpp'...
    'cut_mesh_from_singularities_randomized.cpp', ...
    'polyvector_field_cut_mesh_with_singularities_randomized.cpp',...
    };
% create MATLAB command for building the mex-file
% include necessary directories (e.g. libigl with -I)
matlab_command = sprintf('mex -I%s/include -I%s/external/nanogui/ext/eigen ',LIBIGL_DIR,LIBIGL_DIR);
% then, add the .cpp files necessary
for i = 1:length(cfiles)
    matlab_command = sprintf('%s %s', matlab_command, cfiles{i});
end
% run the MATLAB compilation command - this should create a
% cut_mesh_mex.mexw-whatever filen
eval(matlab_command)

%%           test
[V,F] = load_mesh('data/torus_lo_rt.obj');
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


