clear all; close all; clc;

max_haus = 0.06;

figure('Position',[100 100 992.5 310])
counter = 1;
for i = -0.01:0.005:0.01
    exact_file = sprintf('SAVED_DATA/Hick_Bunny_%0.3f.stl',i);
    sample_file = sprintf('SAVED_DATA/Opt_Mesh_%0.3f.stl',i);
    [vertices1, faces1] = read_ply(exact_file);
    [vertices2, faces2] = read_ply(sample_file);

    bbox_diag = 0.25083813;
    [nearest_indices,~] = knnsearch(vertices1, vertices2);
    hausdorff_distance = sqrt(sum((vertices2 - vertices1(nearest_indices, :)).^2, 2))/bbox_diag;
    
%     max(hausdorff_distance)
%     hausdorff_distances = sort(hausdorff_distance(:));
%     num_values = numel(hausdorff_distances);
%     lower_90_index = round(0.9 * num_values);
%     lower_90_hausdorff_distances = hausdorff_distances(1:lower_90_index);
%     max(lower_90_hausdorff_distances)
    
    figure(1)
    subplot(1,5,counter)
    trisurf(faces2, vertices2(:,1), -vertices2(:,3), vertices2(:,2), hausdorff_distance, 'FaceColor', 'interp', 'EdgeColor', 'none');
    colormap(parula);
    caxis([0, 0.1])
    view(-25,20)
    axis equal off
    xlim([-0.11407467  0.08569895]);
    ylim(-1*[0.07957499 -0.08370341 ]);
    zlim([ 0.01074063  0.21015246]);
    set(gca, 'XTick', [], 'YTick', [], 'ZTick', [])
    set(gcf, 'Color', 'white')
    if counter == 5
        c = colorbar('TickLabelInterpreter', 'latex');
        c.Location = 'eastoutside';
        caxis([0, max_haus])
        c.Position = [0.88 0.2 0.01 0.6];
    end
    sub_pos = get(gca, 'Position');
    sub_pos(1:2) = sub_pos(1:2) - 0.1*ones(1,2);
    sub_pos(3:4) = sub_pos(3:4) + 0.1*ones(1,2);
    set(gca, 'Position', sub_pos);
    switch counter
        case {2,4}
            sub_title = sprintf('%0.3f',i);
        case {1,5}
            sub_title = sprintf('%0.2f',i);
        otherwise
            sub_title = sprintf('%0.1f',i);
    end
    title(sub_title,'FontSize',10,'Interpreter','latex')
    
%     figure
%     trisurf(faces2, vertices2(:,1), -vertices2(:,3), vertices2(:,2), hausdorff_distance, 'FaceColor', 'interp', 'EdgeColor', 'none');
%     colormap(parula);
%     caxis([0, max_haus])
%     if counter == 5
%         c = colorbar;
%         c.Position = [0.9 0.2 0.03 0.6];
%         c.FontSize = 14;
%     end
%     view(-25,20)
%     axis equal off
%     xlim([-0.11407467  0.08569895]);
%     ylim(-1*[0.07957499 -0.08370341 ]);
%     zlim([ 0.01074063  0.21015246]);
%     set(gca, 'XTick', [], 'YTick', [], 'ZTick', [])
%     set(gcf, 'Color', 'white')
%     
%     plot_pos = get(gca, 'Position');
%     set(gca, 'Position', [-0.2 -0.2 1.4 1.4]);
%     
%     set(gcf, 'PaperSize', [8 8]);
%     print(gcf, '-dpdf','-painters','-fillpage', sprintf("PDF_figures/Hausdorff%0.3f.pdf",i))
    
    counter = 1 + counter;
end

figure(1)
set(gcf, 'PaperSize', [6 2]);
% print(gcf, '-dpdf','-bestfit', "PDF_figures/Hausdorff.pdf")
print(gcf, '-dpdf','-painters','-fillpage', "PDF_figures/Hausdorff.pdf")

function [verts,faces] = read_ply(filename)
[stlstruct,~] = stlread(filename);

% Extract the vertices and faces
verts = stlstruct.Points;
faces = stlstruct.ConnectivityList;
end