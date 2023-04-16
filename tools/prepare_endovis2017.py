import os
import sys
import cv2
import json
import numpy as np

# class2sents = {
#     'background': ['background', 'body tissues', 'organs'],
#     'instrument': ['instrument', 'medical instrument', 'tool', 'medical tool'],
#     'shaft': [
#         'shaft', 'instrument shaft', 'tool shaft', 'instrument body',
#         'tool body', 'instrument handle', 'tool handle'
#     ],
#     'wrist': [
#         'wrist', 'instrument wrist', 'tool wrist', 'instrument neck',
#         'tool neck', 'instrument hinge', 'tool hinge'
#     ],
#     'claspers': [
#         'claspers', 'instrument claspers', 'tool claspers', 'instrument head',
#         'tool head'
#     ],
#     'bipolar_forceps': ['bipolar forceps'],
#     'prograsp_forceps': ['prograsp forceps'],
#     'large_needle_driver': ['large needle driver', 'needle driver'],
#     'vessel_sealer': ['vessel sealer'],
#     'grasping_retractor': ['grasping retractor'],
#     'monopolar_curved_scissors': ['monopolar curved scissors'],
#     'other_medical_instruments': [
#         'other instruments', 'other tools', 'other medical instruments',
#         'other medical tools'
#     ],
# }

# gpt-4 v0
class2sents = {
    'background': [
        'background tissue',
        'Background tissue in endoscopic surgery refers to the various normal bodily tissues that surround the surgical site. These tissues often include muscle, fat, connective tissue, blood vessels, and nerves. Colors may vary from pale pink to deep red, depending on the tissue type and blood supply. The shapes are usually irregular and closely packed, with fibrous or soft textures depending on the specific tissue. The function of background tissue is to provide structural support, protect vital organs, and facilitate bodily functions. Characteristics of these tissues include elasticity, varying levels of vascularity, and diverse cellular composition.',
        'In endoscopic surgery, background tissue comprises the normal bodily tissues surrounding the surgical site, such as muscle, fat, connective tissue, blood vessels, and nerves. The colors range from pale pink to deep red, with irregular shapes and varying textures. These tissues serve to support structures, protect organs, and facilitate bodily functions, exhibiting characteristics like elasticity, vascularity, and diverse cellular composition.'
    ],
    'instrument': [
        'surgical instruments',
        'Surgical instruments in endoscopic surgery are specialized tools designed to facilitate minimally invasive procedures. Typically made of stainless steel or other durable materials, they are silver in color. The instruments vary in shape, including long, slender tubes for accessing the surgical site, as well as forceps, scissors, and graspers with various tips for performing specific tasks. Functions of these instruments involve cutting, suturing, manipulating tissues, and cauterizing. Key characteristics include their ergonomic design, precision, and ability to transmit visual and tactile feedback to the surgeon through the endoscope.',
        'Endoscopic surgery instruments are specialized, silver-colored tools made from durable materials like stainless steel. With shapes ranging from slender tubes to forceps, scissors, and graspers, they are designed for cutting, suturing, tissue manipulation, and cauterization. The instruments\' ergonomic design, precision, and ability to provide visual and tactile feedback to the surgeon through the endoscope make them indispensable in minimally invasive procedures.'
    ],
    'shaft': [
        'shaft of surgical instrument',
        'The shaft of a surgical instrument in endoscopic surgery is a slender, elongated component that connects the handle to the working end of the instrument. Usually silver in color, it is made of durable materials like stainless steel or other high-quality metals. The cylindrical shape allows for easy insertion through small incisions or natural body openings. Its function is to facilitate precise control and manipulation of the instrument tip within the surgical site. Key characteristics include its rigidity or flexibility, depending on the procedure, and the ability to transmit force and tactile feedback from the tip to the handle.',
        'In endoscopic surgery, the shaft of a surgical instrument is a slender, elongated, silver-colored component made from durable materials such as stainless steel. Its cylindrical shape enables easy insertion and manipulation within the surgical site, while its rigidity or flexibility and capacity to transmit force and tactile feedback from the tip to the handle ensure precise control during the procedure.'
    ],
    'wrist': [
        'wrist of surgical instrument',
        'The wrist of a surgical instrument in endoscopic surgery is a flexible joint connecting the shaft to the instrument\'s working end. Often silver in color, it is typically made of durable materials such as stainless steel or other high-quality metals. The wrist can have various shapes, depending on the instrument\'s design and specific application, but generally allows for multi-directional movement. Its primary function is to facilitate precise control and angulation of the instrument tip within the surgical site. Key characteristics include its range of motion, stability, and ability to maintain a consistent mechanical advantage for the surgeon.',
        'In endoscopic surgery, the wrist of a surgical instrument is a flexible, silver-colored joint, made from durable materials like stainless steel. Its shape allows for multi-directional movement, and its primary function is to enable precise control and angulation of the instrument tip within the surgical site. The wrist\'s range of motion, stability, and consistent mechanical advantage for the surgeon are crucial characteristics for optimal performance.'
    ],
    'claspers': [
        'claspers of surgical instrument',
        'Claspers of surgical instruments in endoscopic surgery are the grasping or holding components at the working end of the instrument. Generally silver in color, they are made from durable materials such as stainless steel or other high-quality metals. Claspers come in various shapes and sizes, including forceps, graspers, and alligator jaws, depending on the procedure and the tissue being manipulated. Their primary function is to hold, manipulate, or stabilize tissue during surgery. Key characteristics include their precision, the ability to securely grasp without causing trauma, and the capacity to apply controlled pressure.',
        'In endoscopic surgery, claspers are the silver-colored, grasping components at the working end of surgical instruments, made from durable materials like stainless steel. With various shapes like forceps and alligator jaws, claspers are designed to hold, manipulate, or stabilize tissue during surgery. Precision, secure grasping without causing trauma, and the ability to apply controlled pressure are essential characteristics of these components.'
    ],
    'bipolar_forceps': [
        'bipolar forceps',
        'Bipolar forceps in endoscopic surgery are specialized instruments used for coagulation and electrocautery. Typically silver in color, they are made from durable materials such as stainless steel or other high-quality metals. The forceps have two elongated, opposing tips that function as electrodes. Their primary function is to coagulate tissue and control bleeding by applying electrical current between the tips. Key characteristics include their precise application of energy, ability to minimize collateral tissue damage, and the presence of an insulated shaft to prevent unintentional energy transmission.',
        'In endoscopic surgery, bipolar forceps are silver-colored, specialized instruments made from durable materials like stainless steel. Featuring two elongated, opposing tips that serve as electrodes, their primary function is to coagulate tissue and control bleeding using electrical current. Precision in energy application, minimized collateral tissue damage, and an insulated shaft to prevent unintentional energy transmission are essential characteristics of bipolar forceps.'
    ],
    'prograsp_forceps': [
        'prograsp forceps',
        'ProGrasp forceps in endoscopic surgery are versatile grasping instruments designed for secure tissue manipulation. Generally silver in color, they are made from durable materials such as stainless steel or other high-quality metals. The forceps have a unique, multi-fingered tip that resembles a claw or a hand, allowing for a more secure grip on various tissue types. Their primary function is to hold, manipulate, or stabilize tissue during surgery. Key characteristics include their precision, the ability to securely grasp without causing trauma, and the capacity to apply controlled pressure.',
        'In endoscopic surgery, ProGrasp forceps are silver-colored, versatile grasping instruments made from durable materials like stainless steel. With a unique, multi-fingered tip for secure grip, their primary function is to hold, manipulate, or stabilize tissue during surgery. Precision, secure grasping without causing trauma, and the ability to apply controlled pressure are essential characteristics of ProGrasp forceps.'
    ],
    'large_needle_driver': [
        'large needle driver',
        'The large needle driver in endoscopic surgery is a specialized instrument designed for holding and manipulating suturing needles. Typically silver in color, it is made from durable materials such as stainless steel or other high-quality metals. The instrument features a long shaft with a handle at one end and a jaw-like tip at the other end, designed to securely grip the needle. The primary function of a large needle driver is to facilitate suturing and knot tying during surgery. Key characteristics include its precision, secure needle grasp, and ergonomic design for ease of use.',
        'In endoscopic surgery, the large needle driver is a silver-colored, specialized instrument made from durable materials like stainless steel. Featuring a long shaft with a jaw-like tip for securely gripping suturing needles, its primary function is to facilitate suturing and knot tying during surgery. Precision, secure needle grasp, and an ergonomic design for ease of use are essential characteristics of large needle drivers.'
    ],
    'vessel_sealer': [
        'vessel sealer',
        'The vessel sealer in endoscopic surgery is a specialized instrument used for sealing and dividing blood vessels and other soft tissues. Usually silver in color, it is made from durable materials like stainless steel or other high-quality metals. The vessel sealer features a long shaft with a handle at one end and a jaw-like tip at the other end, designed to grasp and seal tissue. Its primary function is to coagulate and cut tissue simultaneously, minimizing bleeding during surgery. Key characteristics include its precision, ability to provide uniform compression, and consistent sealing performance.',
        'In endoscopic surgery, the vessel sealer is a silver-colored, specialized instrument made from durable materials such as stainless steel. With a long shaft and a jaw-like tip for grasping and sealing tissue, its primary function is to coagulate and cut tissue simultaneously, minimizing bleeding. Precision, uniform compression, and consistent sealing performance are essential characteristics of vessel sealers in endoscopic surgery.'
    ],
    'grasping_retractor': [
        'grasping retractor',
        'The grasping retractor in endoscopic surgery is a specialized instrument designed to hold and retract tissues or organs during a procedure. Generally silver in color, it is made from durable materials such as stainless steel or other high-quality metals. The grasping retractor features a long shaft with a handle at one end and a tip with various configurations, such as hooks or claws, at the other end. Its primary function is to expose and maintain visibility of the surgical site. Key characteristics include its precision, secure grasp, and ability to apply controlled pressure without causing tissue damage.',
        'In endoscopic surgery, the grasping retractor is a silver-colored, specialized instrument made from durable materials like stainless steel. With a long shaft and a tip featuring various configurations for holding and retracting tissues, its primary function is to expose and maintain visibility of the surgical site. Precision, secure grasp, and the ability to apply controlled pressure without causing tissue damage are essential characteristics of grasping retractors in endoscopic surgery.'
    ],
    'monopolar_curved_scissors': [
        'monopolar curved scissors',
        'The monopolar curved scissors in endoscopic surgery are specialized instruments designed for cutting and dissecting tissue using electrical energy. Typically silver in color, they are made from durable materials like stainless steel or other high-quality metals. The instrument features a long shaft with a handle at one end and curved scissor-like tips at the other end. Its primary function is to cut and dissect tissue while minimizing bleeding through electrocautery. Key characteristics include precision, controlled cutting, and the ability to coagulate tissue simultaneously.',
        'In endoscopic surgery, monopolar curved scissors are silver-colored, specialized instruments made from durable materials such as stainless steel. Featuring a long shaft with curved scissor-like tips, their primary function is to cut and dissect tissue while minimizing bleeding using electrocautery. Precision, controlled cutting, and the ability to coagulate tissue simultaneously are essential characteristics of monopolar curved scissors in endoscopic surgery.'
    ],
    'other_medical_instruments': [
        'other instruments',
        'Endoscopic surgery also utilizes instruments like trocars, cannulas, and endoscopes. Trocars, usually silver in color and made from durable materials like stainless steel, have a sharp pyramidal or conical tip used to puncture the body cavity. Cannulas are hollow tubes inserted through trocars to maintain the access port. Endoscopes, comprising a long, flexible tube with a camera and light source, enable visualization of the surgical site. These instruments are essential for maintaining access, insufflating the cavity, and providing high-quality visualization during endoscopic surgery.',
        'In endoscopic surgery, additional instruments include silver-colored trocars made from durable materials, featuring a sharp tip for puncturing the body cavity. Cannulas, hollow tubes, maintain the access port, while endoscopes, consisting of a long, flexible tube with a camera and light source, enable visualization of the surgical site. These instruments are vital for maintaining access, insufflating the cavity, and providing high-quality visualization during endoscopic procedures.'
    ],
}

binary_factor = 255
parts_factor = 85
instruments_factor = 32


def get_one_sample(root_dir, image_file, image_path, save_dir, mask,
                   class_name):
    if '.jpg' in image_file:
        suffix = '.jpg'
    elif '.png' in image_file:
        suffix = '.png'
    mask_path = os.path.join(
        save_dir,
        image_file.replace(suffix, '') + '_{}.png'.format(class_name))
    cv2.imwrite(mask_path, mask)
    cris_data = {
        'img_path': image_path.replace(root_dir, ''),
        'mask_path': mask_path.replace(root_dir, ''),
        'num_sents': len(class2sents[class_name]),
        'sents': class2sents[class_name],
    }
    return cris_data


def process(root_dir, cris_data_file):
    cris_data_list = []
    if 'train' in root_dir:
        dataset_num = 8
    elif 'test' in root_dir:
        dataset_num = 10
    for i in range(1, dataset_num + 1):
        image_dir = os.path.join(root_dir, 'instrument_dataset_{}'.format(i),
                                 'images')
        print('process: {} ...'.format(image_dir))
        cris_masks_dir = os.path.join(root_dir,
                                      'instrument_dataset_{}'.format(i),
                                      'cris_masks')
        if not os.path.exists(cris_masks_dir):
            os.makedirs(cris_masks_dir)
        image_files = os.listdir(image_dir)
        image_files.sort()
        for image_file in image_files:
            print(image_file)
            image_path = os.path.join(image_dir, image_file)
            # binary
            binary_mask_file = image_path.replace('images',
                                                  'binary_masks').replace(
                                                      '.jpg', '.png')
            binary_mask = cv2.imread(binary_mask_file)
            binary_mask = (binary_mask / binary_factor).astype(np.uint8)
            for class_id, class_name in enumerate(['background',
                                                   'instrument']):
                target_mask = (binary_mask == class_id) * 255
                if target_mask.sum() != 0:
                    cris_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       cris_masks_dir, target_mask,
                                       class_name))
            # parts
            parts_mask_file = image_path.replace('images',
                                                 'parts_masks').replace(
                                                     '.jpg', '.png')
            parts_mask = cv2.imread(parts_mask_file)
            parts_mask = (parts_mask / parts_factor).astype(np.uint8)
            for class_id, class_name in enumerate(
                ['background', 'shaft', 'wrist', 'claspers']):
                if class_id == 0:
                    continue
                target_mask = (parts_mask == class_id) * 255
                if target_mask.sum() != 0:
                    cris_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       cris_masks_dir, target_mask,
                                       class_name))
            # instruments
            instruments_mask_file = image_path.replace(
                'images', 'instruments_masks').replace('.jpg', '.png')
            instruments_mask = cv2.imread(instruments_mask_file)
            instruments_mask = (instruments_mask / instruments_factor).astype(
                np.uint8)
            for class_id, class_name in enumerate([
                    'background', 'bipolar_forceps', 'prograsp_forceps',
                    'large_needle_driver', 'vessel_sealer',
                    'grasping_retractor', 'monopolar_curved_scissors',
                    'other_medical_instruments'
            ]):
                if class_id == 0:
                    continue
                target_mask = (instruments_mask == class_id) * 255
                if target_mask.sum() != 0:
                    cris_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       cris_masks_dir, target_mask,
                                       class_name))

    with open(os.path.join(root_dir, cris_data_file), 'w') as f:
        json.dump(cris_data_list, f)


if __name__ == '__main__':
    # must add last "/"
    # /jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_test/
    root_dir = sys.argv[1]
    # cris_test.json
    cris_data_file = sys.argv[2]
    process(root_dir, cris_data_file)