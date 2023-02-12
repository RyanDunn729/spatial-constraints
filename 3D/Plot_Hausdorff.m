clear all; close all; clc;

figure('Position',[100 100 1600 600])
counter = 1;
for i = -0.01:0.005:0.01
    subplot(1,5,counter)
    exact_file = sprintf('SAVED_DATA/Hick_Bunny_%0.3f.stl',i);
    sample_file = sprintf('SAVED_DATA/Opt_Mesh_%0.3f.stl',i);
    [vertices1, faces1] = read_ply(exact_file);
    [vertices2, faces2] = read_ply(sample_file);
    bbox_diag = 0.25083813;
    % Calculate the Hausdorff distance
%     hausdorff_distance = HausdorffDist(vertices1, faces1, vertices2, faces2);
    [nearest_indices,~] = knnsearch(vertices1, vertices2);
    hausdorff_distance = sqrt(sum((vertices2 - vertices1(nearest_indices, :)).^2, 2))/bbox_diag;
%     max(hausdorff_distance)
    trisurf(faces2, vertices2(:,1), -vertices2(:,3), vertices2(:,2), hausdorff_distance, 'FaceColor', 'interp', 'EdgeColor', 'none');
    % Add a colorbar to the plot
%     caxis([0, max(hausdorff_distance)]);
%     colorbar;
    view(-25,20)
%     xlabel('x')
%     ylabel('y')
%     zlabel('z')
    axis equal off
    xlim([-0.11407467  0.08569895]);
    ylim(-1*[0.07957499 -0.08370341 ]);
    zlim([ 0.01074063  0.21015246]);
    set(gca, 'XTick', [], 'YTick', [], 'ZTick', [])
    set(gcf, 'Color', 'white')
    sub_pos = get(gca, 'Position')
%     sub_pos(3) = sub_pos(3) * 2;
%     sub_pos(4) = sub_pos(4) * 2;
    set(gca, 'Position', sub_pos);
    counter = 1 + counter;
end
colormap(parula);
c = colorbar;
caxis([0, 0.689])

function [verts,faces] = read_ply(filename)
[stlstruct,~] = stlread(filename);

% Extract the vertices and faces
verts = stlstruct.Points;
faces = stlstruct.ConnectivityList;
end

function hausdorff_distance = HausdorffDist(vertices1, faces1, vertices2, faces2)
    % Compute the distances from each vertex in mesh 1 to mesh 2
    distances1 = min_distance(vertices1, vertices2, faces2);
    
    % Compute the distances from each vertex in mesh 2 to mesh 1
    distances2 = min_distance(vertices2, vertices1, faces1);
    
    % Compute the Hausdorff distance as the maximum of the two distances
    hausdorff_distance = max(max(distances1), max(distances2));
end

function distances = min_distance(vertices1, vertices2, faces2)
    % Initialize the distances
    distances = zeros(size(vertices1, 1), 1);
    
    % Compute the distances from each vertex in mesh 1 to mesh 2
    [nearest_indices, ~] = knnsearch(vertices2, vertices1);
    nearest_vertices = vertices2(nearest_indices, :);
    distances = vecnorm(vertices1 - nearest_vertices, 2, 2);
end